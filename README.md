# AlphaEarth Ili apple-orchard mapping

Runnable Google Earth Engine and Python code for evaluating AlphaEarth satellite embeddings against Sentinel-1/2 and terrain features for cultivated apple-orchard recognition in the Ili River Valley (Xinjiang, China), including weak-label (AOMC) experiments, county-external training expansion, Gongliu polygon holdout evaluation, and a pre-specified Xinyuan one-shot test.

This repository contains **processing and analysis code**. It does not include manuscripts, journal figures, cover letters, AOMC GeoTIFF archives, official Chinese standard boundary packs, or wall-to-wall prediction rasters.

Companion paper (submission target: *Earth Science Informatics*):  
*Weak-label accuracy does not ensure cross-county generalization: An evaluation of AlphaEarth embeddings for apple orchard mapping.*

## Layout

| Path | Contents |
|---|---|
| `gee/` | Earth Engine scripts (ROI, S1/S2 phenology, AlphaEarth, AOMC sampling, annual classify/export, Gongliu visual-interpretation helpers) |
| `python/` | Feature sampling helpers and the revision analysis pipeline |
| `python/revision/` | Numbered experiments (ablation, label efficiency, spatial CV, Gongliu/Xinyuan holdouts, maps, tables) |
| `config/` | Local configuration stubs |
| `data/README.md` | How to obtain public inputs; study-specific data on request |

## Requirements

```bash
python -m pip install -r requirements.txt
```

Earth Engine access needs a Google Cloud / Earth Engine project (the authors used `hallowed-chain-458005-g9` during development; set your own project id). Do not commit credentials.

## Typical entry points

Weak-label / ablation revision pipeline (local metrics once features exist):

```bash
python python/revision/run_revision_pipeline.py --local-only
```

Individual numbered scripts under `python/revision/` (e.g. `01_ablation_b0b6.py`, `32a_poly_pr_metrics_v4.py`, Xinyuan holdout utilities) correspond to the analyses reported in the paper.

GEE scripts in `gee/` are meant to be pasted or imported into the Earth Engine Code Editor / Python API with your project and asset paths updated.

## Data availability

Public satellite and thematic layers are listed under `data/README.md` with provider citations. Study-specific polygon references and model outputs are available from the corresponding author on reasonable request.

## Licence

MIT for the code in this repository. Third-party satellite products, AOMC, and administrative boundaries keep their original terms.
