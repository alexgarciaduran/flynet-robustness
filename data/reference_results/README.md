# Reference results

`R`, `R_std` (bootstrap over trajectories), `LERS` and `flops` per network, one
CSV per condition, from the published run. Rows are in `config.PAPER_NETWORKS`
order. Attacks and perturbations restart from random points, so expect a rerun
to agree to within about one `R_std`.

Two rows differ from Table 2 of the paper:

- sCTRNN is included here, so the transfer numbers average over 13 other
  networks rather than 12.
- RAFT-\*-Sintel under AWN-frames, shot noise and motion jitter: the table
  reports `R ≈ 1.0`, from runs that skipped the ×255 rescaling on the hex
  input. Corrected here: `R ≈ 0.09–0.22` and `≈ 0.57–0.60`. Flicker and fog are
  unchanged.
