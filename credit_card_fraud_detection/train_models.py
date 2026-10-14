"""Credit card fraud detection: Logistic Regression vs Decision Tree vs Random Forest.

Train on fraudTrain.csv, evaluate on fraudTest.csv (later in time, so this is an
out-of-time test). Decision thresholds are tuned for F1 on a time-based validation
slice (last 20% of the training period) so the test set is never used for tuning.
"""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FRAUD_DATA_DIR", HERE / "data"))  # folder with fraudTrain.csv / fraudTest.csv
OUT = HERE


def load():
    usecols = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long",
               "city_pop", "dob", "unix_time", "merch_lat", "merch_long", "is_fraud"]
    tr = pd.read_csv(DATA / "fraudTrain.csv", usecols=usecols)
    te = pd.read_csv(DATA / "fraudTest.csv", usecols=usecols)
    tr["split"], te["split"] = "train", "test"
    return pd.concat([tr, te], ignore_index=True)


def engineer(df):
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df["age"] = (df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["log_amt"] = np.log1p(df["amt"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    # Haversine distance customer -> merchant (km)
    la1, lo1, la2, lo2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    df["dist_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # Per-card behavioural features, using only that card's *earlier* transactions (no label use).
    g = df.groupby("cc_num")
    df["secs_since_prev"] = g["ts"].diff().dt.total_seconds().fillna(1e7).clip(upper=1e7)
    df["log_secs_since_prev"] = np.log1p(df["secs_since_prev"])
    prev_mean = g["amt"].transform(lambda s: s.shift().expanding().mean())
    prev_std = g["amt"].transform(lambda s: s.shift().expanding().std())
    df["amt_vs_card_mean"] = (df["amt"] / prev_mean).fillna(1.0)
    df["amt_zscore_card"] = ((df["amt"] - prev_mean) / prev_std).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 50)
    df["card_txn_idx"] = g.cumcount()
    # Rolling 24h window per card (includes current txn)
    s = df.set_index("ts").groupby("cc_num")["amt"]
    r_cnt = s.rolling("24h").count().reset_index(drop=True)
    r_sum = s.rolling("24h").sum().reset_index(drop=True)
    df["cnt_24h"] = r_cnt.values
    df["log_amt_sum_24h"] = np.log1p(r_sum.values)
    # Same-category amount relative to card's history in that category
    gc = df.groupby(["cc_num", "category"])["amt"]
    cat_prev_mean = gc.transform(lambda s: s.shift().expanding().mean())
    df["amt_vs_card_cat_mean"] = (df["amt"] / cat_prev_mean).fillna(1.0)
    df["gender_m"] = (df["gender"] == "M").astype(int)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


NUM = ["amt", "log_amt", "hour", "dow", "is_night", "age", "log_city_pop", "dist_km", "gender_m",
       "log_secs_since_prev", "amt_vs_card_mean", "amt_zscore_card", "card_txn_idx",
       "cnt_24h", "log_amt_sum_24h", "amt_vs_card_cat_mean"]
CAT = ["category"]


def make_models():
    pre_lr = ColumnTransformer([("num", StandardScaler(), NUM),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pre_tree = ColumnTransformer([("num", "passthrough", NUM),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return {
        "Logistic Regression": Pipeline([("pre", pre_lr), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0))]),
        "Decision Tree": Pipeline([("pre", pre_tree), ("clf", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("pre", pre_tree), ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42))]),
    }


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = np.nanargmax(f1[:-1])
    return float(thr[i])


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(threshold=round(thr, 4), precision=precision_score(y, yhat), recall=recall_score(y, yhat),
                f1=f1_score(y, yhat), pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def main():
    t0 = time.time()
    df = engineer(load())
    assert not df[NUM].isna().any().any(), df[NUM].isna().sum()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"features built in {time.time()-t0:.0f}s; train {len(tr):,} ({tr.is_fraud.mean():.3%} fraud), "
          f"test {len(te):,} ({te.is_fraud.mean():.3%} fraud)")
    cut = int(len(tr) * 0.8)  # tr is time-sorted
    fit_part, val_part = tr.iloc[:cut], tr.iloc[cut:]
    X = NUM + CAT
    results, curves = {}, {}
    for name, model in make_models().items():
        t = time.time()
        model.fit(fit_part[X], fit_part.is_fraud)
        thr = best_f1_threshold(val_part.is_fraud, model.predict_proba(val_part[X])[:, 1])
        model.fit(tr[X], tr.is_fraud)
        p = model.predict_proba(te[X])[:, 1]
        results[name] = {"tuned": metrics(te.is_fraud, p, thr), "default_0.5": metrics(te.is_fraud, p, 0.5)}
        curves[name] = precision_recall_curve(te.is_fraud, p)
        print(f"\n== {name} ({time.time()-t:.0f}s) ==")
        print(json.dumps(results[name], indent=1, default=float))
        print(classification_report(te.is_fraud, (p >= thr).astype(int), digits=4))
        if name == "Random Forest":
            names = model.named_steps["pre"].get_feature_names_out()
            imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
            imp.to_csv(OUT / "rf_feature_importance.csv", header=["importance"])
            print(imp.head(15))

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    rows = [{"model": m, **r["tuned"]} for m, r in results.items()]
    pd.DataFrame(rows).to_csv(OUT / "results_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (prec, rec, _) in curves.items():
        ax.plot(rec, prec, label=f"{name} (PR-AUC {results[name]['tuned']['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall on fraudTest.csv")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "pr_curves.png", dpi=130)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
