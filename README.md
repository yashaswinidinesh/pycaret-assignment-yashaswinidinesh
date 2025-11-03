
# PyCaret Assignment — End‑to‑End Repo

This repo contains **original** Colab‑ready notebooks (no copy/paste from PyCaret examples) covering:

- Classification (binary + multiclass)
- Regression
- Clustering
- Anomaly Detection
- Association Rule Mining (PyCaret 2.3.5 arules)
- Time Series Forecasting (univariate; with/without exogenous)
- **2 Gradio demos** for quick inference from a saved PyCaret pipeline

> **GPU in Colab:** `Runtime → Change runtime type → T4 GPU`. Each notebook passes `use_gpu=True` in `setup()`.  
> If RAPIDS/cuML or GPU‑enabled libs are not available, PyCaret will fall back to CPU automatically.

## How to use
1. Open any notebook in Google Colab.
2. Run the first cell to install the pinned versions.
3. Follow the cells to train, compare/tune, finalize, and save a pipeline (`save_model(...)`).
4. (Optional) Launch a Gradio app locally in Colab for a quick demo.

## Video checklist (≈1 minute each notebook)
- What dataset is used (not from PyCaret examples)
- Target + task, quick EDA check
- `setup(..., use_gpu=True)` + note GPU status
- `compare_models()` → `tune_model()` → `finalize_model()`
- One quick plot (`plot_model(...)`)
- `save_model(...)` + tiny inference
---

**Links referenced in the code comments:**  
- PyCaret GPU docs & notes (use `use_gpu=True`): pycaret.readthedocs.io (classification/clustering/anomaly/time_series APIs)  
- Association rules API (`pycaret.arules` in v2.3.5): pycaret.org/tutorials/html/ARUL101.html  
- Time series with exogenous variables: pycaret.readthedocs.io/en/stable/api/time_series.html  
- RAPIDS/cuML integration background: developer.nvidia.com/blog/streamline-your-model-builds-with-pycaret-rapids-on-nvidia-gpus/

