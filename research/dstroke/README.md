# DStroke research prototype

Implementation of Fu et al. 2021, *Fast Accurate and Automatic Brushstroke Extraction* (ACM TOMM), as a standalone research prototype for the AIAi art-authentication pipeline. Standalone — **not** wired into `src/pictology/*`; lives here until we decide whether to integrate.

## One-time setup

```bash
source C:/Users/MichelleJacobs/Repos/AIAi/.venv/Scripts/activate
# base deps
pip install -r requirements-dstroke.txt
# CUDA PyTorch (adjust cu*** to your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# SAM 2 (strategy C only)
pip install git+https://github.com/facebookresearch/sam2.git
```

Download **`abrupng.exe`** (Windows release) from https://github.com/scurest/abrupng/releases and drop it into this folder (`research/dstroke/`).

SAM 2 checkpoint: follow https://github.com/facebookresearch/sam2#installation — the `sam2.1_hiera_small` variant is enough.

## Run order

| # | Notebook                            | What it produces                                                            |
|---|-------------------------------------|-----------------------------------------------------------------------------|
| 0 | —                                   | **Manual:** curate a stroke-like subset of `brushes/_raw/` into `brushes/curated/` after running `01`'s first cell. |
| 1 | `01_dataset_synthesis.ipynb`        | `data_notcommitted/dstroke_synth/{train,val}/{paintings,edges}/*.png` (500 pairs, 90/10 split) |
| 2a | `02a_pix2pix_finetune.ipynb`       | `outputs/dstroke/strategy_A_pix2pix_finetune.pt` (warm-started modified Pix2Pix) |
| 2b | `02b_pix2pix_from_scratch.ipynb`   | `outputs/dstroke/strategy_B_pix2pix_scratch.pt` (paper-faithful, random init) |
| 2c | `02c_sam2_zero_shot.ipynb`         | `outputs/dstroke/strategy_C_sam2/{val,real}/*_edge.png` (zero-shot SAM 2 baseline) |
| 2d | `02d_segformer_finetune.ipynb`     | `outputs/dstroke/strategy_D_segformer.pt` (non-GAN baseline)                 |
| 3 | `03_strategy_comparison.ipynb`      | IoU/F1/precision/recall table + visual grid; writes winner to `outputs/dstroke/winning_strategy.txt` |
| 4 | `04_soft_segmentation.ipynb`        | `outputs/dstroke/soft/{artist}/{stem}.npz` — per-stroke alpha mattes via guided filter |
| 5 | `05_ordering_and_coloring.ipynb`    | Porter-Duff stroke ordering + recovered per-stroke RGB on overlapping pairs |

Notebooks 2a–2d can be run in any order; they all depend on `01` and are independent of each other. Notebook 3 needs as many of 2a/2b/2d's checkpoints as exist plus 2c's output directory. Notebooks 4 and 5 need 3 to have picked a winner.

## Files

- `dstroke_utils.py` — all shared logic: `extract_abr_brushes`, `load_brush_library`, `synthesize_painting` (Algorithm 1), `render_height_field`, `trapped_ball_merge` (paper §3.2 T(·)), `guided_filter` (He 2010), `porter_duff_ordering` (Algorithm 2), `compute_edge_metrics`, `build_comparison_grid`.
- `brushes/_raw/` — populated automatically by `abrupng`; do not commit.
- `brushes/curated/` — your hand-picked subset; do not commit.
- `requirements-dstroke.txt` — pip requirements (excluding CUDA Torch and sam2; see above).

## Data sources

- Source paintings for synthesis: `data_notcommitted/art500k/toy_dataset.zip` (Web Gallery of Art subset, filtered via `toy_dataset_label.csv` to `FORM == "painting"` and `TECHNIQUE` containing oil / tempera / watercolor / acrylic).
- Real paintings for evaluation: `AIA - AI Tool - Fingerprinting-1/data/raw/{manet,van_gogh,rembrandt}/authenticated/`.
- KyleBrush `.abr` files: `data_notcommitted/{gouache,watercolor,megapack}.abr`.
