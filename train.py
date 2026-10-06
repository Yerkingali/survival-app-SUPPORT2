
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

RANDOM_STATE = 42
URL = "https://raw.githubusercontent.com/autonlab/auton-survival/master/auton_survival/datasets/support2.csv"
OUT = Path("models")
OUT.mkdir(exist_ok=True)

NUM = ["age", "num.co", "scoma", "meanbp", "wblc", "hrt", "resp", "temp",
       "pafi", "alb", "bili", "crea", "sod", "ph", "glucose", "bun"]
CAT = ["sex", "dzgroup", "race"]

# --- Data ---
df = pd.read_csv(URL)
X = df[NUM + CAT].copy()
y = Surv.from_arrays(event=df["death"].astype(bool).values,
                     time=df["d.time"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=df["death"])

# --- Preprocessing (sparse_output=False -> dense matrix, no .toarray() needed) ---
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), CAT),
])
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)

# --- Models ---
cox = CoxPHSurvivalAnalysis(alpha=0.1).fit(Xtr, y_train)
rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=15,
                           max_features="sqrt", n_jobs=-1,
                           random_state=RANDOM_STATE).fit(Xtr, y_train)

# --- Metrics (same as in the notebook) ---
event_times = y_test["time"][y_test["event"]]
low, high = np.percentile(event_times, [10, 90])
grid = np.linspace(low, high, 30)

metrics = {}
for name, m in [("Cox", cox), ("RSF", rsf)]:
    risk = m.predict(Xte)
    c = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    S = np.vstack([fn(grid) for fn in m.predict_survival_function(Xte)])
    ibs = integrated_brier_score(y_train, y_test, S, grid)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, grid)
    metrics[name] = {"C-index ↑": round(float(c), 3),
                     "Mean time-AUC ↑": round(float(mean_auc), 3),
                     "IBS ↓": round(float(ibs), 3)}
    print(name, metrics[name])

# --- Metadata for the app interface ---
meta = {
    "num": NUM,
    "cat": CAT,
    "defaults": {**{c: float(X_train[c].median()) for c in NUM},
                 **{c: str(X_train[c].mode()[0]) for c in CAT}},
    "ranges": {c: [float(X_train[c].quantile(0.01)),
                   float(X_train[c].quantile(0.99))] for c in NUM},
    "categories": {c: sorted(X_train[c].dropna().astype(str).unique().tolist())
                   for c in CAT},
    "feature_names": [n.split("__", 1)[1] for n in pre.get_feature_names_out()],
    "train_mean": Xtr.mean(axis=0).tolist(),     # baseline for explaining Cox
    "test_risk_cox": cox.predict(Xte).tolist(),  # for the risk percentile
    "metrics": metrics,
}

joblib.dump(pre, OUT / "pre.joblib")
joblib.dump(cox, OUT / "cox.joblib")
joblib.dump(rsf, OUT / "rsf.joblib", compress=3)  # RSF is large -> compress
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                              encoding="utf-8")
print("Saved to", OUT.resolve())
