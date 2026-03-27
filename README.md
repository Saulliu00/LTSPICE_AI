# LTspice AI Optimization Pipeline

A modular Python pipeline that uses **LTspice as the simulation engine** and **Optuna as the optimization engine**, with clean interfaces designed for future **PyTorch ML** integration.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [How to Define a New Circuit](#how-to-define-a-new-circuit)
5. [Configuration Reference](#configuration-reference)
6. [Objective Functions](#objective-functions)
7. [Switching Optimization Engines](#switching-optimization-engines)
8. [Plugging in ML Models](#plugging-in-ml-models)
9. [Project Structure](#project-structure)

---

## Architecture

```
config/config.yaml
       │
       ▼
  SearchSpace ──► BaseOptimizer.suggest(trial_id) ──► ParameterSet
                                                            │
                                          ┌─────────────────┘
                                          ▼
                                   NetlistEditor
                                   (writes temp .asc)
                                          │
                                          ▼
                                   LTSpiceRunner
                                   (subprocess batch)
                                          │
                                          ▼
                                   ResultParser
                                   (.raw → SimulationResult)
                                          │
                                          ▼
                                   ObjectiveFunction
                                   (score: float)
                                          │
                          ┌───────────────┤
                          ▼               ▼
               BaseOptimizer        SimulationCache
               .report(score)       (params → result)
                          │
                          ▼
                   PipelineVisualizer
                   (Plotly HTML)
```

**Swap point**: `BaseOptimizer` in `optimization/optimizer.py`.
Both `OptunaOptimizer` and the future `MLOptimizer` implement the same interface. The `Pipeline` loop in `main.py` never changes when you switch engines.

---

## Installation

```bash
pip install -r requirements.txt
```

**LTspice** must be installed separately and the path configured in `config/config.yaml`:

| Platform | Typical path |
|---|---|
| macOS | `/Applications/LTspice.app/Contents/MacOS/LTspice` |
| Windows | `C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe` |
| Linux (Wine) | `wine ~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe` |

---

## Quickstart

### Demo mode (no LTspice required)

Runs the full pipeline using a synthetic analytic RC filter model instead of LTspice:

```bash
python3 main.py --demo
```

### Full run with LTspice

```bash
python3 main.py --config config/config.yaml --trials 50
```

### CLI options

```
--config PATH    Path to YAML config (default: config/config.yaml)
--trials N       Override n_trials from config
--engine ENGINE  Override engine: optuna | random | ml
--demo           Synthetic mode (no LTspice needed)
--no-browser     Don't auto-open plots in browser
```

Output plots are saved to `results/plots/` as interactive HTML files:

| File | Contents |
|---|---|
| `convergence.html` | Score vs trial number |
| `scatter_R1_vs_C1.html` | Parameter space colored by score |
| `frequency_response.html` | Best result's frequency response |
| `dashboard.html` | Combined view of all plots |

---

## How to Define a New Circuit

### Step 1 — Create the LTspice schematic

Open LTspice and draw your circuit. For any component value you want to optimize, replace the numeric value with a parameter reference using curly braces:

```
R1 value: {R1}
C1 value: {C1}
L1 value: {L1}
```

Add a `.param` directive as the default fallback:
```
.param R1=10k C1=100n
```

Save the `.asc` file to `circuits/your_circuit.asc`.

### Step 2 — Update `config.yaml`

```yaml
circuit:
  schematic_path: "circuits/your_circuit.asc"

parameters:
  R1:
    min: 1000.0
    max: 100000.0
    log_scale: true
    type: float
  C1:
    min: 1.0e-9
    max: 1.0e-6
    log_scale: true
    type: float
  L1:
    min: 1.0e-6
    max: 1.0e-3
    log_scale: true
    type: float
```

### Step 3 — Choose the right objective

See [Objective Functions](#objective-functions) below.

### Step 4 — Run

```bash
python3 main.py
```

---

## Configuration Reference

```yaml
ltspice:
  executable: "/Applications/LTspice.app/Contents/MacOS/LTspice"
  timeout: 60         # seconds before subprocess is killed
  retry_count: 3      # retries on transient LTspice crash

circuit:
  schematic_path: "circuits/example_rc_filter.asc"

parameters:
  R1:
    min: 1000.0       # lower bound (Ω)
    max: 1000000.0    # upper bound (Ω)
    log_scale: true   # sample in log space (recommended for R, C, L)
    type: float       # "float" | "int"

optimization:
  engine: optuna      # "optuna" | "random" | "ml"
  n_trials: 50
  direction: minimize # "minimize" | "maximize"
  seed: 42
  sampler: tpe        # "tpe" | "cmaes" | "random"

objective:
  type: cutoff        # see Objective Functions section
  target_signal: "V(out)"
  filter_type: lowpass
  target_cutoff_hz: 1000.0

visualization:
  output_dir: "results/plots"
  show_browser: true  # auto-open plots in default browser
  save_html: true

cache:
  enabled: true
  cache_dir: "results/cache"

logging:
  level: INFO         # DEBUG | INFO | WARNING | ERROR
  log_file: "results/run.log"
```

---

## Objective Functions

All objectives follow the convention: **lower score = better**.

### `type: cutoff` — Low-pass / High-pass filter cutoff frequency

The cutoff frequency **f_c** is the -3 dB point where the output power
drops to half of the passband power:

```
|H(f_c)| = |H_passband| / √2

Equivalently in dB:
|H(f_c)|_dB = |H_passband|_dB − 3 dB
```

Score = `|f_c_actual − f_c_target|` in Hz.

**Config:**
```yaml
objective:
  type: cutoff
  target_signal: "V(out)"
  filter_type: lowpass     # "lowpass" or "highpass"
  target_cutoff_hz: 1000.0
```

- `lowpass`: passband level estimated from the low-frequency end; cutoff is the first frequency where magnitude drops below passband − 3 dB.
- `highpass`: passband level estimated from the high-frequency end; cutoff is the lowest frequency where magnitude rises to within 3 dB of the passband.

---

### `type: bandwidth` — Band-pass filter bandwidth

Bandwidth is the **total width of the passband**, defined as the
difference between the upper and lower -3 dB points:

```
BW = f_high − f_low
```

where **f_low** and **f_high** are the frequencies on the lower and upper
skirts of the passband at which the response falls 3 dB below the peak:

```
|H(f_low)|_dB  = |H_peak|_dB − 3 dB
|H(f_high)|_dB = |H_peak|_dB − 3 dB
```

Score = `|BW_actual − BW_target|` in Hz.

**Config:**
```yaml
objective:
  type: bandwidth
  target_signal: "V(out)"
  target_bw_hz: 500.0     # desired f_high − f_low in Hz
```

---

### `type: mse` — Mean-squared error vs target curve

Minimizes the mean-squared difference between the simulated magnitude
(dB) and a user-supplied target curve.

```yaml
objective:
  type: mse
  target_signal: "V(out)"
  target_values: [0.0, -0.1, -0.5, -3.0, -10.0, -20.0]  # dB
  freq_weights: [1.0, 1.0, 2.0, 3.0, 1.0, 1.0]          # optional
```

---

### `type: peak` — Peak magnitude and frequency

Minimizes combined error in peak magnitude (dB) and peak frequency (Hz):

```
score = |peak_db_actual − peak_db_target| + |log10(f_peak_actual) − log10(f_peak_target)|
```

```yaml
objective:
  type: peak
  target_signal: "V(out)"
  target_peak_db: 3.0       # desired peak magnitude in dB
  target_freq_hz: 5000.0    # desired resonant frequency in Hz
  freq_tolerance_decades: 0.5
```

---

### Writing a custom objective

Subclass `ObjectiveFunction` and register it in the factory:

```python
# core/objective.py

class MyObjective(ObjectiveFunction):
    def __init__(self, target_signal: str, ...):
        self.target_signal = target_signal

    @property
    def name(self) -> str:
        return "my_objective"

    def evaluate(self, result: SimulationResult, params: ParameterSet) -> float:
        sig = result.signals[self.target_signal]
        # ... compute your score ...
        return float(score)

# Add to create_objective() factory:
elif obj_type == "my_objective":
    return MyObjective(signal, ...)
```

---

## Switching Optimization Engines

The only change needed is in `config.yaml`:

```yaml
optimization:
  engine: optuna   # change to "random" or "ml"
```

| Engine | Description | Config keys |
|---|---|---|
| `optuna` | Bayesian optimization (TPE, CMA-ES) | `sampler`, `seed` |
| `random` | Pure random search (baseline) | `seed` |
| `ml` | PyTorch surrogate model *(stub — not yet implemented)* | — |

Or override at runtime:

```bash
python3 main.py --engine random --trials 100 --demo
```

All three engines implement the same `BaseOptimizer` interface in
`optimization/optimizer.py`. The `Pipeline` loop is identical regardless
of which engine is active.

---

## Plugging in ML Models

The `ml/` package is a fully wired stub waiting for a PyTorch implementation.
The integration point is `MLOptimizer` in `ml/trainer.py`.

### What's already done

- `SimulationDataset` (`ml/dataset.py`) — stores `(params, result, score)` tuples and serializes them to disk. Every simulation trial automatically adds to this dataset via `MLOptimizer.report()`.
- `MLOptimizer` (`ml/trainer.py`) — implements `BaseOptimizer`; `suggest()` raises `NotImplementedError` until you fill it in.
- `SurrogateModel` (`ml/surrogate_model.py`) — skeleton feedforward network with correct signature.

### Steps to activate ML

**1. Implement the neural network** in `ml/surrogate_model.py`:

```python
import torch
import torch.nn as nn

class SurrogateModel:
    def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=1):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def predict(self, params: ParameterSet) -> float:
        x = torch.tensor(list(params.values()), dtype=torch.float32)
        with torch.no_grad():
            return float(self.net(x).item())
```

**2. Implement the trainer** in `ml/trainer.py`:

```python
class SurrogateTrainer:
    def train(self, epochs=100):
        X, y = self.dataset.to_tensors()   # implement to_tensors() too
        optimizer = torch.optim.Adam(self.model.net.parameters())
        for _ in range(epochs):
            loss = nn.MSELoss()(self.model.net(X), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**3. Implement `MLOptimizer.suggest()`** in `ml/trainer.py`:

```python
def suggest(self, trial_id: int) -> ParameterSet:
    # Option A: use surrogate model to pick best predicted params
    candidates = [self.search_space.sample_random() for _ in range(1000)]
    scores = [self.model.predict(p) for p in candidates]
    return candidates[int(np.argmin(scores))]
```

**4. Switch engine in config:**

```yaml
optimization:
  engine: ml
```

The pipeline loop in `main.py` requires no changes.

---

## Project Structure

```
LTspice_AI/
│
├── config/
│   └── config.yaml              # All settings (paths, params, objective, etc.)
│
├── core/
│   ├── __init__.py              # Shared types: ParameterSet, SimulationResult, TrialRecord
│   ├── ltspice_runner.py        # Subprocess wrapper for LTspice batch execution
│   ├── netlist_editor.py        # Injects parameter values into .asc / netlist
│   ├── result_parser.py         # Parses .raw output → SimulationResult
│   └── objective.py             # Objective functions (cutoff, bandwidth, mse, peak)
│
├── optimization/
│   ├── optimizer.py             # BaseOptimizer ABC + create_optimizer() factory ← swap point
│   ├── optuna_engine.py         # Optuna (TPE/CMA-ES) + random search implementations
│   └── search_space.py          # ParameterSpec, SearchSpace, log/linear sampling
│
├── ml/                          # PyTorch stubs — implement to activate
│   ├── dataset.py               # SimulationDataset (save/load fully functional)
│   ├── surrogate_model.py       # SurrogateModel skeleton
│   └── trainer.py               # SurrogateTrainer stub + MLOptimizer
│
├── visualization/
│   └── plotly_visualizer.py     # Convergence, scatter, waveform, freq response, dashboard
│
├── utils/
│   ├── logger.py                # setup_logger() with console + file handlers
│   └── cache.py                 # MD5-keyed simulation cache (thread-safe, pickle-based)
│
├── circuits/
│   └── example_rc_filter.asc   # RC low-pass filter with {R1}, {C1} parameters
│
├── main.py                      # Pipeline class + CLI + demo mode (synthetic RC filter)
└── requirements.txt
```
