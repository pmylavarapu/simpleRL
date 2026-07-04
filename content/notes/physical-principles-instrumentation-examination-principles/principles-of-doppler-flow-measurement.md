## The Doppler equation

$$ v = \frac{c \cdot \Delta f}{2 \cdot f_T \cdot \cos\theta} $$

- `v` = blood velocity, `c` = speed of sound in tissue (1540 m/s), `Δf` = Doppler shift, `f_T` = transmitted (transducer) frequency, `θ` = angle between beam and flow.
- The **2** in the denominator reflects the *double* Doppler shift (source → RBC, then RBC → receiver).

## Doppler shift

- Shift = received freq − transmitted freq.
- **Positive** (received > transmitted) when the reflector moves *toward* the transducer.
- **Negative** when reflector moves *away*.
- Intracardiac Doppler shifts fall in the audible range (~20 Hz–20 kHz), even though transducers operate at 2–10 MHz.
- Doppler measures **velocity** (magnitude + direction), not speed.

## Intercept angle

- cos 0° = 1, cos 30° = 0.87, cos 60° = 0.5, cos 90° = 0.
- Beam parallel to flow (0° or 180°) → true velocity is measured.
- Beam perpendicular (90°) → measured velocity = 0.
- Nonparallel angles always **underestimate** velocity.

## CW vs PW Doppler

| | CW | PW |
|---|---|---|
| Crystals | ≥ 2 (one always transmits, one receives) | 1 (alternates) |
| Depth resolution | None (range ambiguity) | Yes (sample-volume specific) |
| Max velocity | Unlimited | Limited by aliasing |
| Use | High velocities (valvular stenosis, regurgitation, TR jet) | Low velocities at a specific site (LVOT, mitral inflow) |

- Maximum unambiguous PW velocity ≈ 1 m/s at ~6 cm depth (varies with depth).
- Simultaneous imaging + Doppler = **duplex** ultrasound.

## Aliasing and the Nyquist limit

- Nyquist limit = **½ × PRF**.
- Aliasing appears when the Doppler shift exceeds the Nyquist limit — the top of the signal "wraps" to the opposite side of the baseline.
- Aliasing can NEVER occur with CW (no PRF constraint).
- **Ways to reduce/eliminate PW aliasing:**
  1. Switch to CW.
  2. Use a lower-frequency transducer (reduces Doppler shift for a given velocity).
  3. Move to a shallower sample volume (raises PRF/Nyquist).
  4. Increase the velocity scale.
  5. Baseline shift (appearance only — doesn't raise the actual limit).
- **High-PRF Doppler:** deliberately places multiple sample gates so signals from twice (or more) the primary depth are recorded simultaneously; extends velocity range at the cost of range ambiguity.

## Sample-volume behavior

- Small sample volume → clean spectral window.
- Large sample volume → spectral broadening (fill-in).
- Sample-volume *depth* is set by the transmit-receive time; sample-volume *length* by the receive-cycle duration.

## Color flow Doppler

- Multi-gate PW Doppler with autocorrelation to estimate **mean** velocity at each location (spectral Doppler reports peak).
- Same PRF / aliasing constraints as PW.
- Standard color map: **red = toward transducer, blue = away**; brightness ∝ velocity up to the Nyquist limit.
- **Variance** (usually green) marks flow disturbance or aliased high-velocity flow.
- Typical **burst length** (packet) = ~8 pulses per scan line — trade-off between velocity accuracy and frame rate.
- To reduce aliasing on color, shift the baseline (allows display up to ~2× the original Nyquist limit).

## Tissue Doppler (TDI)

- Same PW/color hardware, but tuned for the **low-velocity, high-amplitude** motion of myocardium (not RBCs).
- Power output and gain kept low; velocity range small.

## Harmonic imaging

- Second harmonic imaging uses reflections at **2× the transmitted frequency**.
- Improves **lateral resolution by 20–50%**, but degrades axial resolution by 40–100%.
- Makes tissue appear more white; useful in technically difficult studies.

## Mechanical / thermal index

- **MI** = quantifies acoustic pressure (cavitation risk). Lowering MI increases bubble resonance / harmonics (relevant for contrast).
- **TI** = quantifies tissue-heating potential. Target < 1.5 °C tissue heating.
