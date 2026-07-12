# Satellite Image Land-Use Classifier & Temporal Change Detector

ResNet18-based land-use classification (EuroSAT, 10 classes) with
temporal change detection via embedding cosine similarity, GradCAM
explainability, a risk indicator, a rule-based change-explanation
narrative, and a Streamlit dashboard with PDF export.

## Status

Complete. All models are trained with real results, the dashboard is
live, and the full pipeline runs end-to-end:

- **Baseline CNN**: macro-F1 = 0.8999
- **ResNet18 Phase 1 (frozen backbone)**: macro-F1 = 0.7709
- **ResNet18 Phase 2 (fine-tuned, best model)**: macro-F1 = 0.9500
- **UC Merced holdout (mapped subset)**: 17.7% accuracy (domain-gap finding, see report)
- **Change detection ROC AUC**: 0.9902
- **Live dashboard**: https://cei-assignment-bacehmebyzbrsrsabd4vcl.streamlit.app/

See [Satellite_LandUse_Report.pdf](submissions/Satellite_LandUse_Report.pdf) for full results, figures, and discussion.

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
5. **GradCAM is self-implemented, not the `grad-cam` pip package**:
   `src/explainability/gradcam.py` is a from-scratch PyTorch
   implementation (forward/backward hooks on the final conv layer).
   The third-party `pytorch_grad_cam` package was removed entirely
   during deployment, since it has OpenCV hard-baked into its own
   internals and OpenCV was unreliable on the target Python 3.14
   environment. This is why `requirements.txt` does not list
   `grad-cam` — it is intentionally not a dependency, not an omission.

## Bonus Tasks Attempted

| Bonus | Status | Where to see it |
|---|---|---|
| A — GradCAM explainability | ✅ Implemented | `src/explainability/gradcam.py`, dashboard, demo video |
| B — Multi-threshold toggle | ❌ Not attempted | — |
| C — Embedding visualization (t-SNE/UMAP) | ❌ Not attempted | — |
| D — Class-imbalance experiment | ❌ Not attempted | — |

**Additional differentiators (not on the official bonus list, included for presentation quality):**
- Risk indicator (green/yellow/red, derived from the ROC-selected similarity threshold)
- Templated change-explanation narrative (rule-based, explicitly not AI-generated — see dashboard and Section 8.3 of the report)
- Downloadable PDF analysis report, generated on-demand from the dashboard

## Submission

- 📄 Full report: [Satellite_LandUse_Report.pdf](submissions/Satellite_LandUse_Report.pdf)
- 🎥 Demo video: https://youtu.be/DKxYioxKizY
- 🚀 Live dashboard: https://cei-assignment-bacehmebyzbrsrsabd4vcl.streamlit.app/

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
