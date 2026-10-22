"""
@author: alexgarciaduran
"""

import copy
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile


HF_BASE = "https://huggingface.co/alejandrorg00/flynet-robustness/resolve/main/"

# HuggingFace folder -> registry id
HF_DECODERS = {
    "flynet": "FlyNet", "sctrnn": "sCTRNN", "ctrnn": "CTRNN",
    "vcnn_2frames": "vCNN_2_frames", "vcnn_5frames": "vCNN_5_frames",
    "vcnn_10frames": "vCNN_10_frames", "gcnn_2frames": "gCNN_2_frames",
    "raft_small_12": "RAFTsmall", "raft_1": "RAFT_1", "raft_6": "RAFT_6",
    "raft_12": "RAFT_12", "hexraft_1": "RAFT_1_S", "hexraft_6": "RAFT_6_S",
    "hexraft_12": "RAFT_12_S",
}
HF_CHECKPOINTS = {
    "models/cnns/vcnn_2frames/network_000.chkpt": "vCNN_2_frames",
    "models/cnns/vcnn_5frames/network_000": "vCNN_5_frames",
    "models/cnns/vcnn_10frames/network_000": "vCNN_10_frames",
    "models/cnns/gcnn_2frames/network_000.chkpt": "gCNN_2_frames",
}


def _fetch(rel_path, dest, check_only=False):
    """Download one HuggingFace file to `dest`, skipping if it is already there."""
    import urllib.request
    if os.path.exists(dest):
        print("have", dest)
        return dest
    url = HF_BASE + rel_path
    if check_only:
        req = urllib.request.Request(url, method="HEAD")
        size = urllib.request.urlopen(req).headers["content-length"]
        print(f"{size:>10} {rel_path}  ->  {dest}")
        return dest
    print("get ", rel_path, "->", dest, flush=True)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def download_weights(check_only=False):
    """Fetch the CNN/RAFT checkpoints and the decoders from HuggingFace.

    Lays them out where `config.checkpoint_path` and `config.decoder_path`
    look. The flyvis networks and the two vanilla RAFT checkpoints are not
    here -- see the README.
    """
    from flynet_robustness import config

    for hf_dir, net_id in HF_DECODERS.items():
        out = os.path.join(config.DECODER_DIR, net_id, "hit.pkl")
        _fetch(f"decoders/{hf_dir}/hit.pkl", out, check_only)
    for kind in ("position", "velocity"):          # only FlyNet has these
        out = os.path.join(config.DECODER_DIR, "FlyNet", f"{kind}.pth")
        _fetch(f"decoders/flynet/{kind}.pth", out, check_only)

    for rel, net_id in HF_CHECKPOINTS.items():
        _fetch(rel, config.checkpoint_path(net_id), check_only)
    _fetch("models/hexraft_sintel/checkpoint.pth",
           os.path.join(config.RAFT_MODEL_DIR, config.RAFT_HEX_CKPT), check_only)


def install_archive(path):
    """Unpack a flyvis network archive into flyvis.results_dir (or its flow/ subfolder)."""
    from flyvis import results_dir
    names = zipfile.ZipFile(path).namelist() if path.endswith(".zip") else tarfile.open(path).getnames()
    shutil.unpack_archive(path, results_dir if names[0].startswith("flow") else results_dir / "flow")


def synthesize_flynet():
    """Save FlyNet as flow/0000/synthesized_t4d.

    The best ensemble model, with the T4d parameters of the second best, which
    has the correct T4d contrast preference.
    """
    import h5py
    import numpy as np
    import torch
    from flyvis import results_dir
    from flyvis.network import EnsembleView

    loss = "epe" if (results_dir / "flow/0000/000/validation/epe.h5").exists() else "loss"
    ensemble = EnsembleView(results_dir / "flow/0000",
                            best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": loss})
    order = ensemble.argsort(validation_subdir="validation", loss_file_name=loss)
    best_nv = ensemble[order[0]]
    best, second = best_nv.init_network(), ensemble[order[1]].init_network()

    state, donor = copy.deepcopy(best.state_dict()), second.state_dict()
    for i, (_, target) in enumerate(best.edge_params["syn_strength"].keys):
        if target == "T4d":
            state["edges_syn_strength"][i] = donor["edges_syn_strength"][i]
    for key, name in (("nodes_bias", "bias"), ("nodes_time_const", "time_const")):
        i = best.node_params[name].keys.index("T4d")
        state[key][i] = donor[key][i]

    out = results_dir / "flow/0000/synthesized_t4d"
    shutil.rmtree(out, ignore_errors=True)
    (out / "chkpts").mkdir(parents=True)
    (out / "validation").mkdir()
    shutil.copy2(best_nv.dir.path / "_meta.yaml", out / "_meta.yaml")
    torch.save({"network": state}, out / "chkpts" / "chkpt_00000")
    torch.save({"network": state}, out / "best_chkpt")
    for f in ("epe.h5", "loss.h5"):  # single dummy value so the best-checkpoint lookup works
        with h5py.File(out / "validation" / f, "w") as h:
            h.create_dataset("data", data=np.array([0.0]))
    print("Saved FlyNet to", out)


USAGE = """usage: download_assets.py [--weights [--dry-run]] [archive ...]

  --weights   decoders + CNN/RAFT-HexSintel checkpoints from HuggingFace
  --dry-run   with --weights: list sources and destinations, download nothing
  archive...  flyvis network archives to unpack, then synthesize FlyNet
"""

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print(USAGE)
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--weights" in sys.argv:                 # HuggingFace only, no flyvis needed
        download_weights(check_only="--dry-run" in sys.argv)
        sys.exit()
    subprocess.run([sys.executable, "-m", "flyvis", "download-pretrained"], check=True)
    for archive in args:
        install_archive(archive)
    synthesize_flynet()
