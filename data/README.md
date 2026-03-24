# Data

Place client-provided images in the subdirectories below.

```
data/
├── manet/           # Authenticated Manet paintings (TIFF or lossless PNG, ≥300 DPI)
├── contemporary/    # Contemporary artists used as negative class (same format)
└── da_vinci/        # (future) Leonardo da Vinci authenticated works
```

## Naming convention
`{artist}_{title_slug}_{year}.tif` — e.g. `manet_olympia_1865.tif`

## Requirements
- Format: TIFF (preferred, 16-bit) or lossless PNG
- Resolution: ≥ 300 DPI, long edge ≥ 2000 px
- Colour: RGB (do not convert to grayscale)
- Labels: see `data/labels.csv` (create alongside images)
