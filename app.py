"""Streamlit app: survival prediction for seriously ill patients (SUPPORT2).
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS = Path("models")

st.set_page_config(page_title="Patient Survival Predictor", page_icon="🩺", layout="wide")


@st.cache_resource  # load models once, not on every click
def load_artifacts():
    pre = joblib.load(MODELS / "pre.joblib")
    cox = joblib.load(MODELS / "cox.joblib")
    rsf_path = MODELS / "rsf.joblib"
    rsf = joblib.load(rsf_path) if rsf_path.exists() else None
    meta = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
    return pre, cox, rsf, meta


pre, cox, rsf, meta = load_artifacts()

# Human-friendly names for the input fields (left side = column name, do not change)
LABELS = {
    "age": "Age (years)",
    "num.co": "Number of other illnesses",
    "scoma": "Coma score (0 = fully alert, 100 = deep coma)",
    "meanbp": "Mean blood pressure (mmHg)",
    "wblc": "White blood cell count (×1000/µL)",
    "hrt": "Heart rate (beats/min)",
    "resp": "Breathing rate (breaths/min)",
    "temp": "Body temperature (°C)",
    "pafi": "Blood oxygen level (PaO₂/FiO₂)",
    "alb": "Albumin (g/dL)",
    "bili": "Bilirubin (mg/dL)",
    "crea": "Creatinine (mg/dL)",
    "sod": "Sodium (mEq/L)",
    "ph": "Blood pH",
    "glucose": "Blood sugar (mg/dL)",
    "bun": "Blood urea nitrogen (mg/dL)",
    "sex": "Sex",
    "dzgroup": "Main diagnosis",
    "race": "Race / ethnicity",
}


# Short prefixes for categories in charts ("" = show only the value, e.g. "Cirrhosis")
CAT_PREFIX = {"sex": "Sex: ", "race": "Race: ", "dzgroup": ""}

# Glossary of diagnosis groups (keys must match the values in the data)
DIAGNOSES = {
    "ARF/MOSF w/Sepsis": ("Acute Respiratory Failure / Multiple Organ System Failure with Sepsis",
                          "The lungs or several organs stop working properly, "
                          "triggered by a severe body-wide infection (sepsis)."),
    "MOSF w/Malig": ("Multiple Organ System Failure with Malignancy",
                     "Several organs fail at the same time in a patient who also has cancer."),
    "CHF": ("Congestive Heart Failure",
            "The heart is too weak to pump enough blood, so fluid builds up in the lungs and body."),
    "COPD": ("Chronic Obstructive Pulmonary Disease",
             "A long-term lung disease (e.g. emphysema, chronic bronchitis) that makes breathing hard."),
    "Cirrhosis": ("Liver Cirrhosis",
                  "Severe scarring of the liver, so it can no longer do its job properly."),
    "Coma": ("Non-traumatic Coma",
             "Deep unconsciousness not caused by an injury — for example after a stroke "
             "or cardiac arrest."),
    "Colon Cancer": ("Metastatic Colon Cancer",
                     "Bowel cancer that has spread to other organs, most often the liver."),
    "Lung Cancer": ("Advanced Lung Cancer",
                    "Non-small-cell lung cancer at a late stage (III or IV)."),
}


def pretty(feature: str) -> str:
    """Turn model feature names like 'dzgroup_Cirrhosis' into readable text."""
    if feature in LABELS:
        return LABELS[feature].split(" (")[0]
    for col in meta["cat"]:
        prefix = col + "_"
        if feature.startswith(prefix):
            return CAT_PREFIX.get(col, LABELS[col] + ": ") + feature[len(prefix):]
    return feature


# ---------------- Sidebar: patient data ----------------
with st.sidebar:
    st.header("👤 Patient profile")
    st.caption("Fields are pre-filled with typical values. Change any of them to see how the forecast reacts.")
    inputs = {}
    for c in meta["cat"]:
        opts = meta["categories"][c]
        default = meta["defaults"][c]
        inputs[c] = st.selectbox(LABELS.get(c, c), opts,
                                 index=opts.index(default) if default in opts else 0)
    with st.expander("🩸 Vital signs & lab results", expanded=True):
        for c in meta["num"]:
            lo, hi = meta["ranges"][c]
            inputs[c] = st.number_input(
                LABELS.get(c, c), value=round(meta["defaults"][c], 2),
                help=f"Most patients in the data fall between {lo:.1f} and {hi:.1f}.")
    show_rsf = st.toggle("Compare with a second model (Random Survival Forest)",
                         value=rsf is not None, disabled=rsf is None)

X = pd.DataFrame([inputs])[meta["num"] + meta["cat"]]
X[meta["num"]] = X[meta["num"]].astype(float)
Xt = pre.transform(X)

# ---------------- Header ----------------
st.title("🩺 Patient Survival Predictor")
st.markdown(
    "How likely is a seriously ill hospital patient to survive the next month, half-year or year?\n\n"
    "Enter a patient profile on the left and the model estimates the chances, "
    "based on **9,105 real patients** from the SUPPORT2 study."
)
st.caption("⚠️ A data-science portfolio project — not a medical tool and not for clinical decisions.")

tab_pred, tab_why, tab_models, tab_glossary = st.tabs(
    ["📈 Forecast", "🔍 What drives the risk", "📊 How good is the model", "📖 Glossary of main diagnosises"])

# ---------------- Forecast ----------------
with tab_pred:
    models = [("Cox", cox)] + ([("RSF", rsf)] if show_rsf and rsf is not None else [])
    fns = {name: m.predict_survival_function(Xt)[0] for name, m in models}

    # common time grid inside the range of all curves
    t_lo = max(fn.x[0] for fn in fns.values())
    t_hi = min(fn.x[-1] for fn in fns.values())
    MAX_DAYS = 1000  # how far the survival curve is shown
    grid = np.linspace(t_lo, min(t_hi, MAX_DAYS), 200)

    st.subheader("Chance of survival")
    cols = st.columns(3)
    for col, (t, label) in zip(cols, [(30, "1 month"), (180, "6 months"), (365, "1 year")]):
        t = float(np.clip(t, t_lo, t_hi))
        p_cox = float(fns["Cox"](t))
        delta = (f"Second model: {float(fns['RSF'](t)):.0%}" if "RSF" in fns else None)
        col.metric(f"Surviving {label}", f"{p_cox:.0%}", delta=delta, delta_color="off")

    risk = float(cox.predict(Xt)[0])
    pct = float((np.array(meta["test_risk_cox"]) < risk).mean() * 100)
    st.info(f"This patient's risk is higher than for **{pct:.0f}%** of patients in the study.")

    model_names = {"Cox": "Main model (Cox)", "RSF": "Second model (RSF)"}
    curves = pd.concat(
        [pd.DataFrame({"Day": grid.round(0), "Survival": fn(grid), "Model": model_names[name]})
         for name, fn in fns.items()],
        ignore_index=True,
    )
    st.subheader("Survival curve")
    st.caption("Each point shows the chance that the patient is still alive after that many days. "
               "The steeper the curve falls, the higher the risk.")
    curve_chart = (
        alt.Chart(curves)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Day:Q", title="Days after hospital admission",
                    scale=alt.Scale(domain=[0, MAX_DAYS])),
            y=alt.Y("Survival:Q", title="Chance of being alive",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Model:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Day:Q", format=".0f"), "Model:N",
                     alt.Tooltip("Survival:Q", title="Chance of being alive", format=".0%")],
        )
        .properties(height=380)
    )
    st.altair_chart(curve_chart, width="stretch")

# ---------------- Explanation ----------------
with tab_why:
    # For a linear model, SHAP = coef * (x - mean): exact and instant
    contrib = pd.Series(cox.coef_ * (Xt[0] - np.array(meta["train_mean"])),
                        index=[pretty(f) for f in meta["feature_names"]])
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)

    st.subheader("Why this forecast?")
    st.caption("The 10 factors that matter most for **this** patient, compared with an average patient. "
               "Bars to the right **increase** the risk, bars to the left **lower** it.")
    chart_df = pd.DataFrame({"Factor": top.index, "Effect": top.values})
    chart_df["Direction"] = np.where(chart_df["Effect"] > 0, "Increases risk", "Lowers risk")
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Effect:Q", title="Effect on risk"),
            y=alt.Y("Factor:N", sort=None, title=None,           # keep order: most important on top
                    axis=alt.Axis(labelLimit=250)),            # do not cut long names
            color=alt.Color("Direction:N", title=None,
                            scale=alt.Scale(domain=["Increases risk", "Lowers risk"],
                                            range=["#e4572e", "#4c9be8"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["Factor", alt.Tooltip("Effect:Q", format="+.3f")],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("How much does each factor change the risk in general?"):
        st.caption("A **risk multiplier** of 1.5 means 50% higher risk; 0.8 means 20% lower. "
                   "For numbers it is per one typical step up; for categories it is compared "
                   "with the reference group.")
        hr = (pd.DataFrame({"Factor": [pretty(f) for f in meta["feature_names"]],
                            "Risk multiplier": np.exp(cox.coef_)})
              .sort_values("Risk multiplier", ascending=False).reset_index(drop=True))
        st.dataframe(hr.style.format({"Risk multiplier": "{:.2f}"}), width="stretch", hide_index=True)

# ---------------- Model quality ----------------
with tab_models:
    st.subheader("How well do the models work?")
    st.caption("Measured on 2,277 patients the models never saw during training.")
    st.dataframe(pd.DataFrame(meta["metrics"]).T, width="stretch")
    st.markdown(
        "**What the numbers mean**\n"
        "- **C-index** — if you pick two patients at random, how often the model correctly "
        "says who is at higher risk. 0.5 = coin flip, 1.0 = perfect.\n"
        "- **Time-AUC** — the same idea, checked at many points in time.\n"
        "- **IBS** — how far the predicted probabilities are from reality. Lower is better.\n\n"
        "**Which model is used and why**\n"
        "- Both models rank patients by risk equally well.\n"
        "- The second model (Random Survival Forest) is slightly better on average, but it "
        "underestimates the risk for the sickest patients.\n"
        "- The **Cox model** gives more reliable probabilities exactly where it matters most "
        "and is easy to explain — so it is the main model in this app."
    )

# ---------------- Glossary ----------------
with tab_glossary:
    st.subheader("Diagnosis groups explained")
    st.markdown(
        "Every patient in the SUPPORT2 study was admitted to hospital with one of eight serious "
        "conditions. Together they cover the most common reasons why adults end up in intensive "
        "care with a high risk of dying within months. The medical abbreviations below are the "
        "names used in the original dataset; the table translates them into plain language. "
        "The diagnosis is one of the strongest factors in the model: the same age and lab values "
        "can mean very different chances depending on the underlying disease."
    )
    glossary = pd.DataFrame(
        [(code, full, plain) for code, (full, plain) in DIAGNOSES.items()],
        columns=["In the app", "Full name", "What it means"],
    )
    st.dataframe(glossary, width="stretch", hide_index=True)
