# Satellite Image Land-Use Classifier & Temporal Change Detector

ResNet18-based land-use classification (EuroSAT, 10 classes) with
temporal change detection via embedding cosine similarity, GradCAM
explainability, a risk indicator, a rule-based change-explanation
narrative, and a Streamlit dashboard with PDF export.

## Status

All code below has been written and the smoke tests pass (model
builds, forward passes, embedding extraction, cosine similarity, risk
logic, and class mapping all run correctly). **Nothing has been
trained yet** — that requires a GPU session (Colab/Kaggle/local),
which this scaffold doesn't have access to. See "Next Steps" below.

## Setup

```bash
conda create -n satellite python=3.10 -y
conda activate satellite
pip install -r requirements.txt
```

## Datasets

- **EuroSAT**: auto-downloads on first run via `torchvision.datasets.EuroSAT`
  (handled inside `src/data/datasets.py` — no manual step needed).
- **UC Merced Land Use**: manual download required. Get it from the
  original UC Merced Vision Group page, extract so you have:
  ```
  data/raw/ucmerced/<classname>/<image>.tif
  ```

## Running the pipeline (in order)

```bash
# 1. Baseline CNN (time-boxed — ~2-3 hrs including training, don't tune further)
python -m src.training.train_baseline

# 2. Phase 1: frozen backbone
python -m src.training.train_phase1

# 3. Phase 2: unfreeze layer3+layer4, discriminative LR
python -m src.training.train_phase2

# 4. Run the notebooks in notebooks/ for evaluation, spatial leakage,
#    UC Merced holdout, change detection, GradCAM, and error analysis
#    (notebooks are the next thing to build — see Next Steps)

# 5. Launch the dashboard (needs step 3's checkpoint to exist)
streamlit run dashboard/app.py

# 6. Run tests any time
python -m pytest tests/test_smoke.py -v
```

## Rough time estimates (GPU: Colab T4 or better)

| Task | Estimated time |
|---|---|
| EuroSAT download (auto) | 5-10 min |
| UC Merced manual download + extract | 10-20 min |
| Baseline CNN training | 20-40 min |
| Phase 1 (frozen backbone, 3 epochs) | 10-20 min |
| Phase 2 (unfrozen, 5 epochs) | 20-40 min |
| Evaluation notebooks (F1, confusion matrix, UC Merced, leakage) | 1-2 hrs |
| Change detection (pairs, ROC, heatmaps) | 1-2 hrs |
| GradCAM integration/testing | 30-60 min |
| Dashboard testing end-to-end | 1-2 hrs |
| Report writing | 3-4 hrs |
| Demo video recording | 30-60 min |

These are compute + implementation estimates, not including debugging
time — budget extra slack, especially on your first GPU session.

## Architecture

See `config.yaml` for all hyperparameters and paths in one place —
change values there rather than hunting through scripts.

```
src/data/          - dataset loading, transforms, splits, T1/T2 pair simulation
src/models/        - baseline CNN, ResNet18 classifier, embedding extractor
src/training/      - shared train/eval engine + 3 standalone training scripts
src/evaluation/    - metrics, EuroSAT<->UC Merced class mapping, error analysis
src/change_detection/ - cosine similarity, ROC/threshold selection, heatmaps
src/explainability/   - GradCAM
src/transition/       - risk indicator + transition plausibility + explanation
                         (built as ONE module — they share the same inputs)
src/utils/         - PDF report export (fpdf2)
dashboard/         - Streamlit app
tests/             - smoke tests, run before every commit
```

## Documented limitations (state these plainly in your report)

1. **Spatial block split**: the EuroSAT-RGB release used here has no
   real lat/lon metadata per tile. `src/data/splits.py` simulates
   spatial blocks via deterministic filename hashing to preserve the
   methodology (whole regions go entirely to one split) without
   pretending to have ground-truth coordinates.
2. **EuroSAT ↔ UC Merced class mismatch**: 10 vs 21 classes don't map
   1:1. `src/evaluation/class_mapping.py` evaluates only the mapped
   subset; unmapped UC Merced classes are reported separately (or
   skipped entirely if time is tight — see the tiered comments in
   that file).
3. **Change explanation & transition plausibility**: rule-based /
   templated, not a learned or generative model. Don't call this
   "AI-generated" in the report.
4. **Synthetic T1/T2 pairs**: EuroSAT has no real before/after imagery.
   `src/data/region_simulator.py` builds labeled changed/unchanged
   pairs by pairing same-class vs different-class tiles, purely to make
   ROC-curve evaluation possible.

## Next steps (not yet built)

- `notebooks/*.ipynb` — the 8 evaluation notebooks (EDA, ablation
  table, UC Merced eval, spatial leakage, change detection, GradCAM,
  error analysis)
- Actually running training and populating `checkpoints/` with real weights
- Populating `outputs/` with real figures once training has run
- The written report and demo video

## Git setup

```bash
git init
git add .
git commit -m "Initial scaffold: data pipeline, models, training, evaluation, dashboard"
git remote add origin <your-repo-url>
git push -u origin main
```

Checkpoints (`.pt` files) are `.gitignore`d by default since they'll
exceed GitHub's soft 100MB file limit — set up Git LFS if you want them
version-controlled:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```
