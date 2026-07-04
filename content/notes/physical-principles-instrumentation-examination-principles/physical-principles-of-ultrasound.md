## Sound waves

- Sound is a **mechanical, longitudinal** wave (particles vibrate in the direction of propagation). It cannot travel through a vacuum.
- Comprised of compressions (↑ pressure/density) and rarefactions (↓ pressure/density).
- Ultrasound = any wave with frequency **> 20,000 Hz (20 kHz)**. Audible sound = 20 Hz–20 kHz. Infrasound < 20 Hz.

## Basic parameters

| Parameter | Symbol | Units | Determined by | Sonographer adjusts? |
|---|---|---|---|---|
| Period | T | μs | Source | No |
| Frequency | f | Hz (MHz for transducer, kHz for Doppler) | Source | No |
| Wavelength | λ | mm | Source **and** medium | No |
| Propagation speed | c | m/s | Medium only | No |
| Amplitude / power / intensity | — | dB / W / W·cm⁻² | Source (initially) | Yes |

- Period and frequency are reciprocals; higher f → shorter T.
- Frequency and wavelength are inversely related in a given medium.
- Higher f → better axial resolution but less penetration.

## Propagation speed

- Soft-tissue average: **1540 m/s** (= 1.54 mm/μs).
- Depends on medium's density and stiffness. **Stiffness ∝ speed; density ∝ 1/speed.**
- All frequencies travel at the same speed in a given medium.
- Fastest → slowest: **bone (2000–4000) > soft tissue (1540) > fat (1450) > lung (300–1200) > air (330)**. Rule of thumb: solid > liquid > gas.
- Wavelength (mm) in soft tissue = 1.54 / f (MHz).

## Amplitude, power, intensity

- Amplitude = height of the wave (dB).
- Power ∝ amplitude²; **intensity ∝ amplitude²**.
- Doubling US power quadruples intensity.
- dB = 20 · log(A_measured / A_reference).
- 6 dB change = doubling (or halving) of amplitude. 40 dB change = 100× amplitude difference.

## Acoustic impedance

- Z = density × propagation velocity (ρ × c).
- Lung has low density + slow c; bone has high density + fast c.
- Reflection depends on the **difference in acoustic impedance** at an interface.
- Optimal reflection when beam is **perpendicular** to interface.

## Wave–tissue interactions

- **Reflection** — best when perpendicular; if beam is parallel to the interface, "dropout" (little/no reflection back).
- **Scattering** — radiation of ultrasound in multiple directions; occurs with structures smaller than λ (e.g. RBCs). Depends on particle size, hematocrit, transducer frequency, and RBC/plasma compressibility. Produces speckles.
- **Backscatter** (a diffuse form of reflection) is the **major source of information used to build the 2-D image** because it returns energy in many directions.
- **Refraction** — deflection of US at an interface between tissues with different acoustic impedance. Causes double-image artifacts.
- **Attenuation** — loss of signal strength with depth; depends on transducer frequency, tissue attenuation coefficient, distance from transducer, and initial intensity. **Absorption** is the most common mechanism. Lower-frequency transducers penetrate deeper.

## Pulsed ultrasound (from Ch. 2)

- Imaging requires **pulsed** ultrasound. CW cannot form anatomic images (it's used for Doppler).
- A clinical pulse is 2–4 cycles.
- **Pulse duration (PD)** = # cycles × period. Typically 0.5–3 μs. Not changed by sonographer.
- **Spatial pulse length (SPL)** = # cycles × wavelength. Typically 0.1–1 mm. Determined by both source and medium. SPL determines **axial resolution**.
- **Pulse repetition period (PRP)** = start of one pulse to start of next; includes PD + listening time. Determined by imaging depth. Deeper imaging → longer PRP.
- **PRF** = pulses per second (kHz). Inverse of PRP. Determined by imaging depth. **PRF is not the transducer frequency.** As depth ↑, PRF ↓.
- **Duty factor** = fraction of time transmitting. Unitless, typically < 1%. CW = 100% duty factor.

## Range equation / 13 μs rule

- Time for one round trip in soft tissue: **13 μs per cm** of depth.
- PRP (μs) = 13 × depth (cm).
- PRF (Hz) = 77,000 / depth (cm).

## Time-gain compensation (TGC)

- Corrects for attenuation by amplifying returning echoes as a function of depth.
- Default preset: decrease signal in the near field, increase in the far field.

## Aperture and beam

- Aperture = transducer face surface.
- Larger aperture → tighter (more focused) beam; smaller aperture → better angulation.

## Resolution (preview — full detail under I.F)

- Axial resolution — along beam; best. Improved by short SPL (high f, few cycles).
- Lateral resolution — perpendicular to beam; better with focused (narrow) beams.
- Temporal resolution — related to frame rate, PRF.
- Contrast resolution — ability to distinguish tissues of similar echogenicity.

## Bioeffects

- **Thermal** — from absorption of acoustic energy; risk higher with higher intensity, longer dwell time.
- **Non-thermal (mechanical)** — cavitation (gas bubble formation and collapse).
- Guiding principle: **ALARA** (As Low As Reasonably Achievable).
