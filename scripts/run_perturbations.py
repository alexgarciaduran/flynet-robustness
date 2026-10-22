"""
@author: alexgarciaduran
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    p = argparse.ArgumentParser(description="Video-perturbation sweeps -> robustness table.")
    p.add_argument("--networks", nargs="+", default=["all"],
                   help="registry ids, or 'all' for every benchmarked network")
    p.add_argument("--perturbations", nargs="+", default=None,
                   help="subset of AWN_texture AWN_frames Shot_noise Flicker Motion_jitter Fog")
    p.add_argument("--trajectories", type=int, default=None,
                   help="override the per-perturbation trajectory count")
    p.add_argument("--n-trials", type=int, default=10, help="noise draws per trajectory")
    p.add_argument("--results", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--configs", default="data/configs/attack_trajectories.json")
    args = p.parse_args(argv)

    from flynet_robustness import config
    from flynet_robustness.evaluation import PERTURBATIONS, sweep_perturbation
    from flynet_robustness.models import load_network

    names = args.perturbations or list(PERTURBATIONS)
    with open(args.configs, encoding="utf-8") as f:
        trajectories = json.load(f)
    out_dir = os.path.join(args.results, "Environmental")
    os.makedirs(out_dir, exist_ok=True)

    rows = {name: [] for name in names}
    curves = {}

    def dump():
        """Rewrite the tables after each network, so a kill keeps the work."""
        for name, table in rows.items():
            path = os.path.join(out_dir, f"results_Environmental_{name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["network", "R", "R_std", "LERS", "flops"])
                w.writeheader()
                w.writerows(table)
        np.savez(os.path.join(out_dir, "accuracy_curves.npz"), **curves)

    for network_id in config.resolve_networks(args.networks):
        print(f"=== {network_id} ===", flush=True)
        net, decoder = load_network(network_id, device=args.device)
        for name in names:
            r = sweep_perturbation(net, decoder, network_id, name, trajectories,
                                   n_trajectories=args.trajectories, device=args.device,
                                   n_trials=args.n_trials)
            rows[name].append({"network": network_id, "R": r["robustness"],
                               "R_std": r["robustness_std"], "LERS": r["lers"],
                               "flops": config.flops(network_id)})
            curves[f"{name}/{network_id}"] = r["accuracy"]
            curves[f"{name}/magnitudes"] = r["magnitudes"]
            print(f"  {name:14s} R = {r['robustness']:.3f} +/- {r['robustness_std']:.3f}"
                  f"   LERS = {r['lers']:.3f}   ({r['n_trajectories']} trajectories)", flush=True)
        del net, decoder
        if args.device == "cuda":
            torch.cuda.empty_cache()
        dump()

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
