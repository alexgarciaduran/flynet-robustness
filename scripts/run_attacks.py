"""
@author: alexgarciaduran
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PGD_EPS = {"l2": [100, 250, 500, 1000, 2000], "linf": [1, 6, 8, 10, 15]}
CW_C = [1, 10, 1e2, 1e3, 1e4, 1e5]
NES_SIGMA = [1, 3, 6, 9, 12]
TRAJECTORIES = {"pgd": 15, "cw": 10, "nes": 15}


def main(argv=None):
    p = argparse.ArgumentParser(description="PGD / CW / NES on the object texture.")
    p.add_argument("--networks", nargs="+", default=["FlyNet"],
                   help="registry ids, or 'all' for every benchmarked network")
    p.add_argument("--attack", choices=["pgd", "cw", "nes"], default="pgd")
    p.add_argument("--norm", choices=["l2", "linf", "both"], default="both",
                   help="PGD/CW norm")
    p.add_argument("--trajectories", type=int, default=None,
                   help="how many of data/configs/attack_trajectories.json to use")
    p.add_argument("--start-trajectory", type=int, default=0,
                   help="skip this many trajectories, to split a long job into blocks; "
                        "run ids stay globally consistent")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--restarts", type=int, default=2, help="PGD random restarts")
    p.add_argument("--texture", default="Black")
    p.add_argument("--results", default=None, help="output root (sets $FLYNET_RESULTS)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--configs", default="data/configs/attack_trajectories.json")
    args = p.parse_args(argv)

    if args.results:
        os.environ["FLYNET_RESULTS"] = args.results
    from flynet_robustness import config
    if args.results:
        config.RESULTS_DIR = args.results
    from flynet_robustness.attacks import run_cw, run_nes, run_pgd
    from flynet_robustness.models import load_network

    with open(args.configs, encoding="utf-8") as f:
        trajectories = json.load(f)
    n = args.trajectories or TRAJECTORIES[args.attack]
    start = args.start_trajectory
    if start + n > len(trajectories):
        raise SystemExit(f"{args.configs} only has {len(trajectories)} trajectories "
                         f"(asked for {start}..{start + n})")
    norms = ["l2", "linf"] if args.norm == "both" else [args.norm]
    stim = [dict(config.DEFAULT_STIMULUS, **t, device=args.device,
                 a=1, mode="avg_pool_adaptive") for t in trajectories[start:start + n]]

    for network_id in config.resolve_networks(args.networks):
        print(f"=== {network_id} ({args.attack}) ===", flush=True)
        net, decoder = load_network(network_id, device=args.device)
        for offset, base_kwargs in enumerate(stim):
            i = start + offset
            torch.manual_seed(2 * (i + 1))
            if args.attack == "pgd":
                for norm in norms:
                    for eps in PGD_EPS[norm]:
                        run_pgd(net, decoder, network_id, norm=norm, eps=eps, lr=eps / 10,
                                steps=args.steps or 100, n_restarts=args.restarts,
                                base_kwargs=base_kwargs, texttype=args.texture,
                                run_id=i + 1, device=args.device)
            elif args.attack == "cw":
                for c in CW_C:
                    for norm in norms:
                        run_cw(net, decoder, network_id, norm=norm, c=c,
                               steps=args.steps or 150, lr=0.1, base_kwargs=base_kwargs,
                               texttype=args.texture, run_id=i + 1, device=args.device)
            else:
                for sigma in NES_SIGMA:
                    run_nes(net, decoder, network_id, sigma=sigma, steps=args.steps or 100,
                            n_samples=15, lr=1.0, base_kwargs=base_kwargs,
                            texttype=args.texture, run_id=i + 1, device=args.device)
        del net, decoder
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
