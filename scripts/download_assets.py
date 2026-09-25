"""Set up the flyvis networks.

    python scripts/download_assets.py [0003.tar.gz ctrnn_edge_matched.zip flow_0301_best10.zip]

1. downloads the flyvis pretrained ensemble (flow/0000);
2. unpacks the given network archives into the flyvis results folder
   (CTRNN flow/0003, sCTRNN flow/0215_pruned_current_ft_compact, degree-matched flow/0301_best10);
3. synthesizes FlyNet (flow/0000/synthesized_t4d): the best flow/0000 model with the
   T4d parameters (input synapse strengths, bias, time constant) of the 2nd best, which
   has the correct T4d ON contrast preference.
CNN/RAFT checkpoints and decoders are placed manually (checkpoints/README.md, decoders/README.md).
"""

import copy
import shutil
import subprocess
import sys
import tarfile
import zipfile


def install_archive(path):
    """Unpack a flyvis network archive into flyvis.results_dir (or its flow/ subfolder)."""
    from flyvis import results_dir
    names = zipfile.ZipFile(path).namelist() if path.endswith(".zip") else tarfile.open(path).getnames()
    shutil.unpack_archive(path, results_dir if names[0].startswith("flow") else results_dir / "flow")


def synthesize_flynet():
    """Save flow/0000/synthesized_t4d (FlyNet)."""
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


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "flyvis", "download-pretrained"], check=True)
    for archive in sys.argv[1:]:
        install_archive(archive)
    synthesize_flynet()
