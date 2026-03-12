# Learning-Augmented AoI Scheduling

**Training a Fast Neural Policy from an Integer-Programming Oracle**

> SYSC 5804 / ELG 6184: 5G Networks (Winter 2026)

## Setup

```bash
git clone https://github.com/Hzz-Git/Age-of-Information-Scheduling.git
cd Age-of-Information-Scheduling
pip install -r requirements.txt
```

Requires Python 3.10+.

## Reproducing Results

All commands are run from the repo root. All experiments use **seed 42**.

### Step 1: Train the neural policy (~5 min on CPU)

```bash
python experiments/aoi/run_training.py
```

Generates oracle demonstrations (MIP, H=5, T=2000 slots), trains two models (BC and perturbed), and saves weights + normalization stats to `experiments/aoi/output/training/`.

### Step 2: Nominal 7-policy comparison (Table I)

```bash
python experiments/aoi/run_nominal.py
```

Evaluates Uniform, Round-Robin, Max-AoI, LP+Round, MIP Oracle, Neural BC, and Neural Pert under nominal conditions (T=1000).

### Step 3: Oracle horizon ablation

```bash
python experiments/aoi/run_sweeps.py
```

Sweeps H ∈ {1, 3, 5, 10, 15} to confirm AoI saturates at H=3.

### Step 4: Delivery model mismatch (2x2 experiment)

```bash
python experiments/aoi/run_mismatch_test.py
```

Runs both policies under both the true (exponential) and matched (linear) simulators to verify the mismatch hypothesis.

### Step 5: Distribution shift robustness

```bash
python experiments/aoi/run_shift_experiments.py
```

Tests all policies (without retraining) under arrival burst, channel degradation, and combined shift at ρ ∈ {0, 0.1, 0.3, 0.5, 0.8, 1.0}.

## Project Structure

```
src/aoi/
├── config.py            # Simulation parameters (N=10, K=10, M=3)
├── simulator.py         # AoI environment (exponential delivery)
├── simulator_linear.py  # Matched simulator (linear delivery)
├── oracle.py            # MIP oracle (receding-horizon, HiGHS)
├── baselines.py         # Uniform, Round-Robin, Max-AoI, LP+Round
├── policy.py            # MLP policy (30 → 128 → 64 → 10)
├── dataset.py           # Feature extraction + perturbation
├── train.py             # Behavioral cloning training loop
└── metrics.py           # AoI, P95, violation rate, TTL drops

experiments/aoi/
├── run_training.py          # Step 1: data generation + training
├── run_nominal.py           # Step 2: 7-policy comparison
├── run_sweeps.py            # Step 3: horizon sweep
├── run_mismatch_test.py     # Step 4: 2×2 mismatch ablation
└── run_shift_experiments.py # Step 5: distribution shift

configs/aoi/
└── default.yaml             # Default experiment parameters
```

## License

MIT
