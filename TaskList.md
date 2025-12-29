# Project Task List — Milestones

This task list converts the Project Plan "Project Milestones" into actionable tasks for implementation and tracking.

- [x] Milestone 1 — Project skeleton, input parsing, `yfinance` fetcher, caching (2 days)
  - [x] Create project skeleton and folders
  - [x] Implement input parser (`input_parser.py`)
  - [x] Implement data fetcher with caching (`data_fetcher.py`)
  - [x] Add `data_store/` cache folder placeholder
  - [x] Add `requirements.txt`, `README.md`, and basic tests

- [x] Milestone 2 — Preprocessing & feature engineering (3 days)
  - [x] Standardize timeline and align tickers
  - [x] Handle missing data (forward/backward fill implemented)
  - [x] Handle corporate actions (split-adjustment heuristic implemented)
  - [x] Implement per-asset features: returns (1d, 5d, 20d), rolling vol, momentum, drawdown
  - [x] Implement cross-sectional ranks
  - [x] Unit tests for feature correctness

- [x] Milestone 3 — `PortfolioEnv` + baseline agent (5 days)
  - [x] Design Gym-compatible `PortfolioEnv` observation & action spaces (skeleton implemented)
  - [x] Implement reward (risk-adjusted / Sharpe-like) and step logic
  - [x] Implement transaction costs and rebalancing mechanics
  - [x] Add unit tests for reward and step calculations
  - [x] Add a simple baseline agent (random or heuristic)

- [ ] Milestone 4 — Integrate `stable-baselines3` trainer (5 days)
  - [ ] Choose algorithm (PPO/SAC) and config-driven training harness (`train_agent.py`)
  - [ ] Add checkpointing & logging
  - [ ] Support vectorized envs and seeds for reproducibility
  - [ ] Basic hyperparameter configuration and example runs

- [ ] Milestone 5 — Monte Carlo simulator and plotting (3 days)
  - [ ] Calibrate GBM params (mu, sigma) from historical log-returns
  - [ ] Estimate correlation and implement correlated sampling (Cholesky)
  - [ ] Simulate M=30 portfolio paths and produce plots
  - [ ] Produce percentile, VaR, CVaR and terminal distribution summaries
  - [ ] Tests for calibration & simulation consistency

- [ ] Milestone 6 — Recommendation engine & per-asset analytics (3 days)
  - [ ] Compute asset-level metrics: contribution to return/vol, correlation, momentum
  - [ ] Combine signals into score for "increase" / "decrease" exposure
  - [ ] Output ranked add/reduce lists with rationales and suggested weight deltas
  - [ ] Unit tests for scoring logic

- [ ] Milestone 7 — Notebook / Streamlit UI (2 days)
  - [ ] Create `notebooks/demo.ipynb` with upload, run Monte Carlo, and show plots
  - [ ] (Optional) Build a Streamlit app for interactive use
  - [ ] Side-by-side comparison view (user vs RL allocation)

- [ ] Milestone 8 — Packaging, docs, tests, and handoff (2 days)
  - [ ] Finalize `README.md` and quickstart instructions
  - [ ] Add CI (basic `pytest` workflow) and packaging notes
  - [ ] Run full test-suite and prepare handoff materials

## Notes & Next Actions
- Prioritize Milestone 2 after completing the baseline skeleton.
- Assign owners, set dates, and break larger milestones into smaller PR-sized tasks when ready.

Progress update (2025-12-29): Milestone 3 subtasks completed (env design, reward/step logic, transaction-cost handling, tests, baseline agent). Next: begin Milestone 4 — integrate `stable-baselines3` trainer and training harness.


