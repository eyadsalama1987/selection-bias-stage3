diff --git a/README.md b/README.md
index e69de29bb2d1d6434b8b29ae775ad8c2e48c5391..cf4c423a77fa6680717d4e273e5373b8fef0883a 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,44 @@
+# Selection Bias Stage 3 (Importance Weighting)
+
+This project implements **Importance Weighting** to address sample selection bias on the UCI Cleveland Heart Disease dataset. It reproduces multiple bias levels, compares three models, and outputs tables/plots for analysis.
+
+## Requirements
+
+- Python 3.9+
+
+Install dependencies:
+
+```bash
+pip install -r requirements.txt
+```
+
+## Run the experiment
+
+Run everything end-to-end (download data, create bias, train models, and save outputs):
+
+```bash
+python -m src.run_experiments
+```
+
+## Outputs
+
+After running, you will find (generated locally, not committed):
+
+- `data/heart.csv`: cleaned dataset.
+- `outputs/results.csv`: metrics for each bias level.
+- `outputs/performance.png`: accuracy and F1 vs bias level.
+- `outputs/weights.png`: importance weight distribution.
+- `outputs/data_bias_scatter.png`: PCA scatter of biased train vs unbiased test.
+- `report/stage3_report.md`: final report for stage 3.
+
+## Project structure
+
+```
+src/
+  data.py           # data download/cleaning/splitting/bias
+  weighting.py      # domain classifier + weights + clipping + ESS
+  plot.py           # figures
+  run_experiments.py
+outputs/
+report/
+```
