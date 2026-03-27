# Circuit Design Reference

Reference knowledge for each supported circuit topology, including transfer functions, key parameters, design equations, and how to map them to this pipeline's objective functions.

---

## Table of Contents

1. [Low-Pass Filter (LPF)](#1-low-pass-filter-lpf)
2. [High-Pass Filter (HPF)](#2-high-pass-filter-hpf)
3. [Band-Pass Filter (BPF)](#3-band-pass-filter-bpf)
4. [Notch Filter (Band-Stop)](#4-notch-filter-band-stop)
5. [DC-DC Buck Converter](#5-dc-dc-buck-converter)
6. [Class-E Amplifier](#6-class-e-amplifier)

---

## 1. Low-Pass Filter (LPF)

### What it does
Passes signals below the cutoff frequency and attenuates signals above it. Used for anti-aliasing, noise rejection, and audio bass filtering.

### First-order RC (passive)

```
Vin ──[R]──┬── Vout
           [C]
           │
          GND
```

Transfer function:

```
H(s) = 1 / (1 + s·R·C)
```

Magnitude response:

```
|H(jω)| = 1 / √(1 + (ω/ω_c)²)
```

#### Key parameters

| Parameter | Formula | Description |
|---|---|---|
| Cutoff frequency | f_c = 1 / (2π·R·C) | -3 dB point; output power = ½ input power |
| -3 dB condition | \|H(f_c)\| = 1/√2 ≈ 0.707 | Magnitude drops 3 dB below passband |
| Roll-off (1st order) | −20 dB/decade | Rate of attenuation above f_c |
| Phase at f_c | −45° | |

#### Second-order RLC (passive)

```
Vin ──[L]──[R]──┬── Vout
                [C]
                │
               GND
```

Transfer function:

```
H(s) = ω_0² / (s² + (ω_0/Q)·s + ω_0²)
```

| Parameter | Formula |
|---|---|
| Natural frequency | ω_0 = 1/√(L·C) |
| Quality factor | Q = (1/R)·√(L/C) |
| Roll-off (2nd order) | −40 dB/decade |

#### Pipeline objective
```yaml
objective:
  type: cutoff
  filter_type: lowpass
  target_cutoff_hz: 1000.0
```

#### Design tip
To set a target f_c: choose C first (standard value), then solve `R = 1/(2π·f_c·C)`. Use log-scale parameter search for both R and C.

---

## 2. High-Pass Filter (HPF)

### What it does
Passes signals above the cutoff frequency and attenuates signals below it. Used for DC blocking, AC coupling, and audio treble filtering.

### First-order RC (passive)

```
Vin ──[C]──┬── Vout
           [R]
           │
          GND
```

Transfer function:

```
H(s) = s·R·C / (1 + s·R·C)
```

Magnitude response:

```
|H(jω)| = (ω/ω_c) / √(1 + (ω/ω_c)²)
```

#### Key parameters

| Parameter | Formula | Description |
|---|---|---|
| Cutoff frequency | f_c = 1 / (2π·R·C) | -3 dB point; output power = ½ input power |
| -3 dB condition | \|H(f_c)\| = 1/√2 ≈ 0.707 | Passband is at high frequency end |
| Roll-off (1st order) | +20 dB/decade below f_c | |
| Phase at f_c | +45° | |

#### Second-order RLC (passive)

```
Vin ──[C]──[R]──┬── Vout
                [L]
                │
               GND
```

Transfer function:

```
H(s) = s² / (s² + (ω_0/Q)·s + ω_0²)
```

#### Pipeline objective
```yaml
objective:
  type: cutoff
  filter_type: highpass
  target_cutoff_hz: 1000.0
```

#### Design tip
HPF and LPF share the same f_c formula. For a HPF, swap R and C positions relative to LPF. The passband level for HPF is estimated at high frequencies; the pipeline's `CutoffObjective` automatically handles this when `filter_type: highpass` is set.

---

## 3. Band-Pass Filter (BPF)

### What it does
Passes signals within a frequency band and attenuates those outside it. Used in RF receivers, audio equalizers, and sensor signal conditioning.

### Series RLC (passive)

```
Vin ──[R]──[L]──[C]──┬── Vout
                      [R_load]
                      │
                     GND
```

Transfer function (voltage across R_load):

```
H(s) = (ω_0/Q)·s / (s² + (ω_0/Q)·s + ω_0²)
```

Magnitude response peaks at ω_0 with value 1 (0 dB for matched load).

#### Key parameters

| Parameter | Formula | Description |
|---|---|---|
| Center frequency | f_0 = 1/(2π·√(L·C)) | Frequency of peak response |
| Quality factor | Q = (1/R)·√(L/C) = f_0/BW | Selectivity; higher Q = narrower band |
| Bandwidth | BW = f_high − f_low = f_0/Q | Total width between -3 dB points |
| Lower -3 dB point | f_low = f_0·(√(1 + 1/(4Q²)) − 1/(2Q)) | |
| Upper -3 dB point | f_high = f_0·(√(1 + 1/(4Q²)) + 1/(2Q)) | |
| Geometric mean | f_0 = √(f_low · f_high) | Center is geometric (not arithmetic) mean |

#### -3 dB definition in this pipeline
Both f_low and f_high are where:
```
|H(f)|_dB = |H_peak|_dB − 3 dB
```
Bandwidth is:
```
BW = f_high − f_low
```

#### Pipeline objective
```yaml
objective:
  type: bandwidth
  target_signal: "V(out)"
  target_bw_hz: 200.0
```

#### Design tip
For a target f_0 and BW: `L = Q·R/(2π·f_0)` and `C = 1/(4π²·f_0²·L)`. Narrow BW (high Q) requires tight component tolerances.

---

## 4. Notch Filter (Band-Stop)

### What it does
Attenuates a specific narrow frequency band while passing all others. Used to eliminate a known interference (e.g., 50/60 Hz mains hum, switching noise).

### Twin-T RC notch (passive)

```
         [R]──[R]
Vin ─┬───┤         ├─── Vout
     │   [C]──[C]  │
     │      │      │
     │     [2C]   [R/2]
     │      │      │
     └──────┴──────┴─── GND
```

Transfer function:

```
H(s) = (s² + ω_n²) / (s² + 4·ω_n·s + ω_n²)
```

where `ω_n = 1/(R·C)`.

### Series LC notch (in shunt path)

```
Vin ──[R_series]──┬── Vout
                 [L]
                 [C]   ← series LC to ground; short-circuits at f_0
                  │
                 GND
```

At resonance, the LC branch is a short circuit, pulling Vout to 0.

#### Key parameters

| Parameter | Formula | Description |
|---|---|---|
| Notch frequency | f_n = 1/(2π·R·C) or 1/(2π·√(L·C)) | Frequency of maximum attenuation |
| Attenuation depth | Ideally −∞ dB; practical: −20 to −60 dB | |
| Q factor | Controls notch width; higher Q = narrower notch | |
| Stopband width | BW_stop = f_n / Q | |

#### Pipeline objective
Use `peak` objective inverted, or `mse` with a target that is 0 dB everywhere except the notch frequency:
```yaml
objective:
  type: mse
  target_signal: "V(out)"
  target_values: [0, 0, 0, -40, 0, 0, 0]   # dip at notch frequency
```
Or use `peak` to minimize the notch depth error:
```yaml
objective:
  type: peak
  target_signal: "V(out)"
  target_peak_db: -40.0       # desired attenuation at notch
  target_freq_hz: 60.0        # 60 Hz mains notch
```

#### Design tip
The Twin-T is sensitive to component matching. A 1% mismatch can reduce attenuation depth significantly. In LTspice, use `.param` to sweep R and C symmetrically.

---

## 5. DC-DC Buck Converter

### What it does
Steps down a DC voltage with high efficiency using a switching transistor, inductor, and capacitor. Output voltage is controlled by the duty cycle D.

### Basic topology

```
         SW (MOSFET)
Vin ──┬──[SW]──┬── Vout ──[R_load]── GND
      │        [L]    │
      │        │      [C_out]
     [D1]      │      │
      │        └──────┘
     GND
```

### Steady-state relationships (continuous conduction mode, CCM)

```
Vout = D · Vin

Iout = Vout / R_load

Duty cycle:  D = Vout / Vin   (0 < D < 1)
```

#### Key parameters

| Parameter | Formula | Description |
|---|---|---|
| Output voltage | V_out = D · V_in | Controlled by duty cycle |
| Inductor ripple current | ΔI_L = (V_in − V_out)·D / (f_sw·L) | Peak-to-peak ripple in L |
| Output voltage ripple | ΔV_out = ΔI_L / (8·f_sw·C_out) | Peak-to-peak ripple at output |
| Critical inductance (CCM boundary) | L_c = (1−D)·R_load / (2·f_sw) | Below this, enters DCM |
| Efficiency (ideal) | η = 1 (100%) | Real losses in SW, D, L, C, R |

#### Design rules of thumb

| Goal | Rule |
|---|---|
| Low output ripple | Large C_out, high f_sw, large L |
| Fast transient response | Small L (lower impedance), high f_sw |
| High efficiency | Low R_DS(on) MOSFET, low-ESR capacitor, Schottky diode |
| Stable CCM operation | L > L_c at minimum load |

#### LTspice simulation directives
```spice
.param D=0.5 Fsw=100k
.tran 0 5m 4m 10n     ; transient: run 5ms, save last 1ms
```

#### Pipeline objective
Use `mse` comparing the output voltage waveform against a flat DC target, or optimize for output ripple:
```yaml
objective:
  type: mse
  target_signal: "V(out)"
  # target_values: flat array at desired Vout level
```

#### Design tip
Start simulation in steady state by running enough cycles for the inductor current to reach equilibrium. Use `.ic` or run sufficient transient time (at least 5× L/R time constant) before measuring.

---

## 6. Class-E Amplifier

### What it does
A switching power amplifier optimized for high efficiency at RF frequencies (typically 1–300 MHz). The transistor switches between fully ON and fully OFF, with waveform shaping ensuring zero-voltage switching (ZVS) and zero-voltage-derivative switching (ZVDS) to eliminate switching losses.

### Basic topology

```
              RFC (choke)
Vdd ──[L_choke]──┬── Vdrain
                 │
                [SW]  (MOSFET / BJT)
                 │
                GND

Vdrain ──[C_shunt]──GND         (shunt capacitor, often Coss of MOSFET)

Vdrain ──[L_series]──[C_series]──[R_load]──GND   (series resonant output network)
```

### Zero-voltage switching (ZVS) conditions

At the moment the switch turns ON, both the drain voltage and its derivative must be zero:

```
V_drain(t_on) = 0          (zero voltage switching, ZVS)
dV_drain/dt|_{t=t_on} = 0  (zero dV/dt switching, ZVDS)
```

These conditions eliminate capacitive discharge losses at turn-on, enabling near-100% efficiency.

### Ideal Class-E design equations (50% duty cycle, optimum operation)

| Parameter | Formula | Notes |
|---|---|---|
| Shunt capacitance | C_shunt = 0.1836 / (ω · R_load) | ω = 2π·f_sw |
| Series inductance | L_series = Q_L · R_load / ω | Q_L typically 5–10 |
| Series capacitance | C_series = 1 / (ω · R_load · (Q_L − 0.2116)) | Tunes series branch to resonance |
| RF choke | L_choke ≫ L_series | Presents high impedance at f_sw; practical: ≥ 10× L_series |
| Output power | P_out = 0.5768 · V_dd² / R_load | |
| Peak drain voltage | V_peak ≈ 3.56 · V_dd | Must be within MOSFET V_DS(max) |
| Peak drain current | I_peak ≈ 2.86 · I_dc | Must be within MOSFET I_D(max) |
| Theoretical efficiency | η = 100% (ideal) | Practical: 85–95% |

### Key design constraints

| Constraint | Implication |
|---|---|
| V_DS(max) > 3.56 · V_dd | Choose MOSFET with sufficient breakdown voltage |
| ZVS condition met | Correct C_shunt is critical; too small → hard switching |
| High Q_L resonator | Better harmonic rejection, narrower bandwidth |
| Duty cycle ≈ 50% | Standard Class-E; adjustable for different loading |

### LTspice simulation directives
```spice
.param Vdd=5 Fsw=13.56Meg RL=50
.tran 0 10u 8u 1n        ; run 10µs, observe last 2µs (steady state)
.four {Fsw} V(out)        ; Fourier analysis at switching frequency
```

### Pipeline objective
Use `peak` to maximize output power at the fundamental frequency, or `mse` to shape the output waveform:
```yaml
objective:
  type: peak
  target_signal: "V(out)"
  target_peak_db: 20.0       # desired output magnitude (dBV)
  target_freq_hz: 13560000   # 13.56 MHz ISM band example
```

#### Design tip
Class-E is very sensitive to component values — small deviations break ZVS. When using this pipeline, set tight parameter bounds (±20% of calculated values) and use a high-Q resonator (`Q_L ≥ 7`) to maintain waveform shaping. Monitor `V_drain` waveform to verify it touches zero before switch turn-on.
