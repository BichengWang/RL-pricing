# R1 input and L1 probe evidence

Fetched the ICAIF 2020 FinRL repository at `5dbb44dfcb8a01a94762dc44fca2bcb2c626d1dc`, hashing all 184 files, including its committed `done_data.csv` and source CSVs. Also fetched Yahoo daily chart responses for `^DJI` and DIA, 2009-01-02 through 2025-12-31. URLs, versions, byte sizes, SHA-256 hashes and UTC retrieval times are in [`hashes.json`](hashes.json); rerun [`fetch_inputs.py`](fetch_inputs.py) to refresh them. Raw downloads stay in the ignored `.cache/r1-finrl-ensemble/` directory.

On 2026-10-25, the [L1 GitHub Actions probe](https://github.com/BichengWang/RL-pricing/actions/runs/40218954659) passed on Python 3.7 with TensorFlow 1.15.4 and Stable-Baselines 2.10.1. It imported the authors' `run_DRL.py` and trained A2C for five steps on the pinned data. **Recommendation: GO for the full L1 run.** This smoke establishes that the pinned environment builds; it does not establish reproduction of the paper's reported metrics.

P0.19 moves the L1 go/no-go from 2026-11-17 to 2026-11-05.
