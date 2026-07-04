## Beam structure

- Ultrasound beam has three regions: **near field (Fresnel zone)** → **focal zone** (narrowest) → **far field (Fraunhofer zone)** where the beam diverges.
- **Lateral resolution is best in the focal zone**, degrades in the far field as the beam widens.
- Larger aperture → tighter focal beam (better lateral resolution).
- Focusing (electronic or mechanical) narrows the beam at the depth of interest.

## Resolution (types)

- **Axial (longitudinal) resolution** — distance between two reflectors along the beam axis. Determined by spatial pulse length (SPL). Shorter SPL (higher f, fewer cycles) = better. Best of the three types.
- **Lateral resolution** — distance perpendicular to beam. Determined by beam width; best at the focal zone.
- **Temporal resolution** — related to frame rate. Better with narrow sector, shallow depth, fewer focal zones.
- **Contrast resolution** — ability to distinguish tissues of different reflectivity.
- **Spatial resolution** — line density; write-zoom increases it, read-zoom does not.

## Frame rate trade-offs

- Frame rate ↑ with narrow sector, shallow depth, single focal zone, low line density.
- Real-time 2D: ~30–100 Hz.

## Zoom

- **Write zoom** — reacquires with more scan lines / pixels in the ROI → truly improves image resolution.
- **Read zoom** — magnifies the acquired image (no resolution gain).

## Imaging artifacts (major types)

| Artifact | Mechanism | Recognition / fix |
|---|---|---|
| **Reverberation** | Multiple back-and-forth reflections between two strong parallel reflectors | Equally spaced, parallel lines at ↓ intensity with depth. Common with prosthetic valves. |
| **Comet-tail** | Reverberation from small metallic/highly reflective object | Solid hyperechoic beam distal to object; lines are not equidistant. Change to harmonic imaging helps. |
| **Ring-down** | Reverberation with strong repetitive ringing of the crystal | Similar to comet-tail but different mechanism. Often gas particles. |
| **Mirror image** | Structure in front of a highly reflective surface duplicates deeper | Duplicated structure equidistant beyond the reflector (violates straight-line assumption). |
| **Side lobe** | Off-axis energy from array transducer strikes a strong reflector | Image appears displaced laterally from true location. |
| **Beam width** | Structure at the edge of the beam superimposes on the central image | E.g., aortic valve "in" LA, atheroma "in" aortic lumen. Adjust focal zone. |
| **Refraction** | Beam bends at an interface with different propagation speeds (Snell's law) | Lateral displacement or duplication (e.g., double AV in short axis). |
| **Range ambiguity** | Echoes from deep structures return after next pulse fires | Increase depth so echo returns before next pulse. |
| **Acoustic shadowing** | Strong reflector attenuates sound beyond it | Anechoic zone distal to prosthetic valve, calcification. |
| **Near-field clutter ("bang" artifact)** | High-amplitude ringing of crystal in near field | Higher-frequency transducer, harmonic imaging, decrease depth. |
| **Focal enhancement (banding)** | Horizontal band of echoes at focal zone | Adjust focal zone. |
| **Ghosting** | Multiple reflections on color Doppler | Color extends beyond anatomic borders. |
| **Propagation-speed error** | US speed in tissue differs from assumed 1540 m/s | Distorts apparent depth (e.g., silastic ball of Starr-Edwards). |

### Artifact spatial relationship

- **More distant than object:** reverberation (parallel motion) or mirror image (opposite motion).
- **Same distance as object:** beam width or side lobe.

## Doppler-specific artifacts

- **Mirror-image (crosstalk):** symmetric spectrum on both sides of baseline; caused by high Doppler gain — reduce gain.
- **Range ambiguity in Doppler:** signals from > one depth mixed (high-PRF Doppler).
- **Aliasing:** wrap-around above Nyquist.
- **Beam-width Doppler artifact:** overlap of signals from adjacent flows.

## Harmonic imaging

- Second-harmonic imaging uses reflections at 2× the transmitted frequency.
- Reduces **side lobes, grating lobes, reverberations, near-field clutter**.
- Improves lateral resolution 20–50%; worsens axial resolution 40–100%.
