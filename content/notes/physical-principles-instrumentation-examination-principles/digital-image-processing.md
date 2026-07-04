## Pre- vs post-processing

- **Preprocessing** — applied before storage: gain, TGC, dynamic range compression, filtering. Changes what is stored.
- **Postprocessing** — applied after storage: gray-scale map, edge enhancement, colorization. Does NOT change stored data; can be reapplied non-destructively.

## Dynamic range

- Range of signal amplitudes (from smallest to largest) represented in the display.
- **Lower dynamic range → higher contrast** (black-white image, fewer gray shades).
- **Higher dynamic range → smoother, more gray-shade tissue characterization** (softer image).

## Frame averaging (persistence)

- Averages consecutive frames to reduce noise.
- Trade-off: reduced temporal resolution (blurs rapid motion).

## Scan conversion

- Converts polar (radial) beam-line data into a rectangular pixel grid for the display.
- Interpolation fills in pixels between sparse scan lines.

## Zoom types

- **Write zoom** — reacquires the ROI at higher line density / pixel density → true resolution gain.
- **Read zoom** — magnifies stored data (no resolution gain).

## Image storage / DICOM

- Modern echo systems store as DICOM files (metadata + image or cine loop).
- Compression may be lossy (JPEG) or lossless.

## Frame rate and depth relationship

- Frame rate ↑ with: narrow sector, shallow depth, few focal zones, low line density.
