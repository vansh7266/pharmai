# ── PharmAI Environment Version Checker ──────────────────────
# Run this in your pharmai/ folder: python check_versions.py

import sys

print("=" * 55)
print("  PharmAI — Environment Version Report")
print("=" * 55)
print(f"  Python         : {sys.version.split()[0]}")

libs = [
    ("numpy",       "numpy"),
    ("pandas",       "pandas"),
    ("scikit-learn", "sklearn"),
    ("xgboost",      "xgboost"),
    ("fastapi",      "fastapi"),
    ("uvicorn",      "uvicorn"),
    ("keras",        "keras"),
    ("tensorflow",   "tensorflow"),
    ("scipy",        "scipy"),
    ("shap",         "shap"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
]

for display_name, import_name in libs:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed (no version attr)")
        print(f"  {display_name:<16}: {ver}")
    except ImportError:
        print(f"  {display_name:<16}: ❌ NOT INSTALLED")

print("=" * 55)

# Also verify model files exist
import os
print("\n  Model files check:")
required = [
    "models/xgb_model.pkl", "models/rf_model.pkl", "models/gb_model.pkl",
    "models/scaler.pkl",    "models/ts_scaler.pkl", "models/ae_scaler.pkl",
    "models/lstm_model.keras", "models/autoencoder.keras",
    "models/ae_threshold.npy",
    "models/ensemble_metrics.json", "models/maintenance_report.json", "models/carbon_summary.json"
]
all_ok = True
for f in required:
    exists = os.path.exists(f)
    kb = os.path.getsize(f)/1024 if exists else 0
    icon = "✅" if exists else "❌"
    print(f"  {icon}  {f:<45} {'%.1f KB' % kb if exists else 'MISSING'}")
    if not exists: all_ok = False

print()
if all_ok:
    print("  ✅ All model files present — ready to run!")
else:
    print("  ❌ Some files missing — re-run notebook Cell 25")
print("=" * 55)
