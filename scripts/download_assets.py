"""Fetch / locate model assets.

FlyNet and CTRNN weights come from flyvis. CNN/RAFT checkpoints and trained
decoders are placed manually (see checkpoints/README.md, decoders/README.md).
"""

import subprocess
import sys


def download_flyvis_pretrained():
    """Download flyvis pretrained networks (FlyNet / CTRNN)."""
    subprocess.run([sys.executable, "-m", "flyvis", "download-pretrained"], check=True)


if __name__ == "__main__":
    print("Downloading flyvis pretrained models (FlyNet / CTRNN)...")
    try:
        download_flyvis_pretrained()
    except Exception as e:  # noqa: BLE001
        print("flyvis download failed:", e)
        print("Run `flyvis download-pretrained` manually, then place CNN/RAFT "
              "checkpoints and decoders per the READMEs.")
