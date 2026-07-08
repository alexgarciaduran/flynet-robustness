"""Differentiable 3D looming-stimulus pipeline (PyTorch3D).

Builds textured primitives, renders a linear looming trajectory batched over
time, and converts frames to luminance downsampled to model input size.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import ico_sphere

from .textures import get_texture

# --- Renderer ---------------------------------------------------------------


def get_renderer(image_size, device_str, bg=(0.5, 0.5, 0.5),
                 faces_per_pixel=1, sigma=1e-4, gamma=1e-4):
    """Ambient-only Phong renderer with a fixed camera at the origin."""
    device = torch.device(device_str)
    cameras = FoVPerspectiveCameras(
        device=device, R=torch.eye(3)[None].to(device),
        T=torch.zeros(1, 3).to(device), fov=60.0, znear=0.1, zfar=100.0)
    lights = PointLights(device=device, ambient_color=((1, 1, 1),),
                         diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
    blend = BlendParams(sigma=sigma, gamma=gamma, background_color=bg)
    rast = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                 faces_per_pixel=faces_per_pixel)
    shader = SoftPhongShader(cameras=cameras, lights=lights,
                             blend_params=blend, materials=Materials(device=device))
    return MeshRenderer(MeshRasterizer(cameras=cameras, raster_settings=rast), shader)


# --- Primitive meshes -------------------------------------------------------


def _double_sided(faces):
    return torch.cat([faces, faces.flip(dims=[1])], dim=0)


def create_square(device="cpu", half_size=1.0, double_sided=True):
    device = torch.device(device)
    verts = torch.tensor([[-half_size, -half_size, 0.0], [half_size, -half_size, 0.0],
                          [half_size, half_size, 0.0], [-half_size, half_size, 0.0]],
                         dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device)
    if double_sided:
        faces = _double_sided(faces)
    verts_uvs = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]],
                             dtype=torch.float32, device=device)
    return verts, faces, verts_uvs, faces.clone()


def create_disk(device="cpu", radius=1.0, n_segments=64, double_sided=True):
    device = torch.device(device)
    center = torch.zeros(1, 3, device=device)
    angles = torch.linspace(0, 2 * np.pi, n_segments + 1, device=device)[:-1]
    ring = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles),
                        torch.zeros_like(angles)], dim=1)
    verts = torch.cat([center, ring], dim=0)
    idx = torch.arange(1, n_segments + 1, device=device, dtype=torch.int64)
    faces = torch.stack([torch.zeros_like(idx), idx, torch.roll(idx, -1, 0)], dim=1)
    if double_sided:
        faces = _double_sided(faces)
    xy = verts[:, :2] / radius
    verts_uvs = torch.stack([0.5 * (xy[:, 0] + 1), 0.5 * (xy[:, 1] + 1)], dim=1).clamp(0, 1)
    return verts, faces, verts_uvs, faces.clone()


def create_sphere(device="cpu", level=3):
    sphere = ico_sphere(level=level, device=device)
    verts, faces = sphere.verts_list()[0], sphere.faces_list()[0]
    r = torch.norm(verts, dim=1).clamp_min(1e-12)
    u = (0.5 + torch.atan2(verts[:, 2], verts[:, 0]) / (2 * np.pi)).frac()
    v = 0.5 - torch.asin(torch.clamp(verts[:, 1] / r, -1.0, 1.0)) / np.pi
    uv_v = torch.stack([u, v], dim=1)
    n = faces.shape[0]
    uv = uv_v[faces.reshape(-1)].clone()
    faces_uvs = torch.arange(3 * n, device=device, dtype=torch.int64).view(n, 3)
    u_face = uv[:, 0].view(n, 3)
    crosses = (u_face.max(1).values - u_face.min(1).values) > 0.5
    uv[:, 0] = (u_face + (crosses[:, None] & (u_face < 0.5)).float()).view(-1)
    return verts, faces, uv, faces_uvs, True


def create_cube(device="cpu"):
    verts = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
                         dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 2, 4], [4, 2, 6], [0, 4, 1], [1, 4, 5], [0, 1, 2], [2, 1, 3],
                          [7, 5, 3], [3, 5, 1], [7, 3, 6], [6, 3, 2], [7, 6, 5], [5, 6, 4]],
                         dtype=torch.int64, device=device)
    verts_uvs = torch.tensor(
        [[0.25, 0.75], [0.5, 0.75], [0.5, 1.0], [0.25, 1.0],
         [0.25, 0.25], [0.5, 0.25], [0.5, 0.5], [0.25, 0.5],
         [0.25, 0.0], [0.5, 0.0], [0.5, 0.25], [0.25, 0.25],
         [0.25, 0.5], [0.5, 0.5], [0.5, 0.75], [0.25, 0.75],
         [0.5, 0.25], [0.75, 0.25], [0.75, 0.5], [0.5, 0.5],
         [0.0, 0.25], [0.25, 0.25], [0.25, 0.5], [0.0, 0.5]],
        dtype=torch.float32, device=device)
    faces_uvs = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                              [8, 10, 9], [8, 11, 10], [12, 14, 13], [12, 15, 14],
                              [16, 18, 17], [16, 19, 18], [20, 22, 21], [20, 23, 22]],
                             dtype=torch.int64, device=device)
    return verts, faces, verts_uvs, faces_uvs


_MESH2D = {"Disk": create_disk, "Square": create_square}


def build_stimulus(shape, rotation, texture, device="cpu"):
    """Textured mesh for a given shape, in-plane/axis rotation and texture."""
    device = torch.device(device)
    if texture is None:
        texture = get_texture("Black")
    texture = texture.to(device=device, dtype=torch.float32).clamp(0, 1)

    needs_tile_u = False
    if shape == "Sphere":
        verts, faces, verts_uvs, faces_uvs, needs_tile_u = create_sphere(device)
    elif shape == "Cube":
        verts, faces, verts_uvs, faces_uvs = create_cube(device)
    elif shape in _MESH2D:
        verts, faces, verts_uvs, faces_uvs = _MESH2D[shape](device=device)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    axis = torch.tensor([[0.0, 0.0, 1.0]] if shape in _MESH2D else [[0.0, 1.0, 0.0]],
                        dtype=torch.float32, device=device)
    R = axis_angle_to_matrix(axis * torch.tensor([rotation], device=device))[0]
    verts = verts @ R.T

    if needs_tile_u:
        maps = texture.repeat(1, 1, 2, 1)
        verts_uvs = verts_uvs.clone()
        verts_uvs[:, 0] = verts_uvs[:, 0] / 2.0
    else:
        maps = texture
    tex_uv = TexturesUV(maps=maps.repeat(1, 1, 1, 3),
                        faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
    return Meshes(verts=[verts], faces=[faces], textures=tex_uv)


# --- Trajectory -------------------------------------------------------------


def sample_start(radius, jitter_radius=0.0, theta=None):
    """XY start point on a circle of given radius."""
    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi)
    x = radius * np.cos(theta) + np.random.uniform(-jitter_radius, jitter_radius)
    y = radius * np.sin(theta) + np.random.uniform(-jitter_radius, jitter_radius)
    return np.array([x, y])


def compute_trajectory(start_xy, z0=15.0, z_end=2.0, hit=True, extra_radius=1.0):
    """Start position and direction vector for a hit or near-miss trajectory."""
    x0, y0 = start_xy
    if hit:
        target = np.array([0.0, 0.0, z_end])
    else:
        angle = np.arctan2(y0, x0) + np.pi
        target = np.array([-0.25 * x0 + extra_radius * np.cos(angle),
                           -0.25 * y0 + extra_radius * np.sin(angle), z_end])
    start_pos = np.array([x0, y0, z0])
    return start_pos, target - start_pos


def compute_angular_size_trajectory(start_pos, dir_vec, R=1.0, T=1.0, dt=0.01,
                                    sphere=False, ntp=None):
    """Angular size and angular speed of the object along the trajectory."""
    times = np.linspace(0, T, ntp) if ntp else np.arange(0, T, dt)
    pos = start_pos[None, :] + np.outer(times / T, dir_vec)
    D = np.linalg.norm(pos, axis=1)
    ang = 2 * np.arcsin(np.clip(R / D, 0, 1)) if sphere else 2 * np.arctan(R / D)
    return times, D, ang, np.gradient(ang, dt)


# --- Recording (luminance + downsampling) -----------------------------------


def img_2vis(img, out_size=416, mode="bilinear_antialiased"):
    """Convert rendered RGB(A) frames (N,H,W,C) to luminance (N,out,out)."""
    assert img.ndim == 4 and img.shape[-1] in (3, 4)
    rgb = img[..., :3]
    Y = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).clamp(0, 1)
    Y4 = Y.unsqueeze(1)
    if mode == "bilinear_antialiased":
        Yrs = F.interpolate(Y4, size=(out_size, out_size), mode="bilinear",
                            align_corners=False, antialias=True)
    elif mode == "avg_pool_adaptive":
        Yrs = F.adaptive_avg_pool2d(Y4, (out_size, out_size))
    else:
        raise ValueError("mode must be 'bilinear_antialiased' or 'avg_pool_adaptive'")
    return Yrs.squeeze(1).clamp(0, 1)


def recording(mesh, renderer, n_frames, start_pos, dir_vec, device="cpu",
              out_size=416, mode="bilinear_antialiased", return_rgb=False):
    """Render all timepoints in one batched call; return luminance and positions."""
    device = torch.device(device)
    v0 = mesh.verts_list()[0].to(device)
    f0 = mesh.faces_list()[0].to(device)
    tex0 = mesh.textures.to(device)

    ts = (torch.zeros(1, device=device) if n_frames <= 1
          else torch.linspace(0, 1, n_frames, device=device))
    start = torch.tensor(start_pos, dtype=torch.float32, device=device)
    direction = torch.tensor(dir_vec, dtype=torch.float32, device=device)
    pos = start[None, :] + ts[:, None] * direction[None, :]
    position = pos.detach().cpu().T

    verts_all = v0.unsqueeze(0) + pos.unsqueeze(1)
    meshes = Meshes(verts=[verts_all[i] for i in range(n_frames)],
                    faces=[f0 for _ in range(n_frames)], textures=tex0.extend(n_frames))
    imgs = renderer(meshes)
    vis = img_2vis(imgs, out_size=out_size, mode=mode)
    if return_rgb:
        return vis, position, imgs[..., :3]
    return vis, position


def generate_looming_stimulus(hit=False, radius=5.0, theta=0.0, shape="Sphere",
                              texture=None, time=2.0, fps=100, stop_margin_hit=0.1,
                              device="cpu", z0=15.0, out_size=416, a=1.0, rotation=0.0,
                              jitter_radius=0.0, mode="bilinear_antialiased",
                              return_rgb=False, renderer=None):
    """Generate a looming sequence and its kinematics.

    Returns (vis, speed, angle, position); with return_rgb also the RGB frames.
    """
    device = str(device)
    n_frames = int(round(time * fps))
    stimulus = build_stimulus(shape, rotation, texture, device=device)
    render_size = max(1, int(round(float(a) * float(out_size))))
    if renderer is None:
        renderer = get_renderer(render_size, device, bg=(0.5, 0.5, 0.5))

    start_xy = sample_start(radius, jitter_radius, theta)
    bounding_radius = stimulus.verts_padded()[0].norm(dim=1).max().item()
    z_end = float(bounding_radius + 0.1 + stop_margin_hit)
    start_pos, dir_vec = compute_trajectory(start_xy, z0=z0, z_end=z_end, hit=hit)

    out = recording(stimulus, renderer, n_frames, start_pos, dir_vec, device=device,
                    out_size=out_size, mode=mode, return_rgb=return_rgb)
    if return_rgb:
        vis, position, frames = out
    else:
        vis, position = out
        frames = None

    path_len = float(np.linalg.norm(dir_vec))
    speed = path_len / float(time) if time > 0 else path_len / max(1, n_frames - 1)
    lateral = float(np.linalg.norm(start_xy)) * (1.0 if hit else 1.5)
    angle = float(np.arctan2(lateral, max(1e-6, z0 - z_end)))

    if return_rgb:
        return vis, frames, speed, angle, position
    return vis, speed, angle, position


def save_frames_as_gif(frames, path="output.gif", fps=30):
    """Save a [T,H,W(,C)] tensor in [0,1] as a looping GIF."""
    import imageio.v2 as imageio
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (frames.detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.mimsave(path, arr, duration=1 / fps, loop=0)
