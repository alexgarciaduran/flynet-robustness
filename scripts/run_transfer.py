"""
@author: alexgarciaduran
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PGD_EPS = {"l2": [100, 250, 500, 1000, 2000], "linf": [1, 6, 8, 10, 15]}


def main(argv=None):
    p = argparse.ArgumentParser(description="Each network's PGD textures, applied to the others.")
    p.add_argument("--networks", nargs="+", default=["all"])
    p.add_argument("--norm", choices=["l2", "linf", "both"], default="both")
    p.add_argument("--attacks", type=int, default=15, help="PGD runs per epsilon")
    p.add_argument("--texture", default="Black")
    p.add_argument("--pgd-root", default=None,
                   help="where the PGD runs live (default: $FLYNET_RESULTS)")
    p.add_argument("--results", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.pgd_root:
        os.environ["FLYNET_RESULTS"] = args.pgd_root
    from flynet_robustness import config
    if args.pgd_root:
        config.RESULTS_DIR = args.pgd_root
    from flynet_robustness.attacks.transfer import save_transfer, transfer_matrix
    from flynet_robustness.metrics import bootstrap_robustness, lers, transfer_auc_matrix
    from flynet_robustness.models import load_networks

    ids = config.resolve_networks(args.networks)
    norms = ["l2", "linf"] if args.norm == "both" else [args.norm]
    out_dir = os.path.join(args.results, "Transfer")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {len(ids)} networks onto {args.device} ...", flush=True)
    nets = load_networks(ids, device=args.device)

    for norm in norms:
        eps_list = PGD_EPS[norm]
        ids_out, acc = transfer_matrix(nets, eps_list, norm=norm, attacks=args.attacks,
                                       texttype=args.texture, device=args.device,
                                       network_ids=ids)
        save_transfer(os.path.join(out_dir, f"transfer_{norm}.pkl"), ids_out, acc, eps_list, norm)

        rows = []
        for j, target in enumerate(ids_out):
            # this target under every other network's textures
            per_source = np.moveaxis(acc[:, j], 0, -1).reshape(len(eps_list), -1)
            per_source = per_source[:, ~np.isnan(per_source).any(axis=0)]
            if per_source.size == 0:
                print(f"  {target}: no PGD textures found, skipped")
                continue
            R, R_std = bootstrap_robustness(eps_list, per_source)
            rows.append({"network": target, "R": R, "R_std": R_std,
                         "LERS": lers(R, target), "flops": config.flops(target)})
            print(f"  {target:18s} transfer R = {R:.3f} +/- {R_std:.3f}", flush=True)

        path = os.path.join(out_dir, f"results_Transfer_{norm}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["network", "R", "R_std", "LERS", "flops"])
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)
        np.save(os.path.join(out_dir, f"auc_matrix_{norm}.npy"),
                transfer_auc_matrix(acc, eps_list))


if __name__ == "__main__":
    main()
