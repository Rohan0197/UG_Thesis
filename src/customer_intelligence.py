import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, silhouette_score
import warnings

warnings.filterwarnings("ignore")

RAW = Path("raw_data")
OUT = Path("processed_data")
OUT.mkdir(exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
DISCOUNT_RATE = 0.10  # annual discount rate for CLV calculation
CHURN_THRESHOLD_DAYS = 90  # days since last purchase → labelled as churned
N_CLUSTERS = 4  # K-Means segments (validated by elbow + silhouette)

print("=" * 55)
print("Step 4: Customer Intelligence (UCI Dataset)")
print("=" * 55)

# ── 1. Load and combine UCI data ──────────────────────────────────────────────
print("\nLoading UCI Online Retail data...")
df_II = pd.read_excel(RAW / "online_retail_II.xlsx", sheet_name=0)
df_I = pd.read_excel(RAW / "Online Retail.xlsx")
df_I = df_I.rename(
    columns={"CustomerID": "Customer ID", "InvoiceNo": "Invoice", "UnitPrice": "Price"}
)
df = pd.concat([df_II, df_I], ignore_index=True)

# ── 2. Clean ──────────────────────────────────────────────────────────────────
print("Cleaning data...")
df = df.dropna(subset=["Customer ID"])
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["Customer ID"] = df["Customer ID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["revenue"] = df["Quantity"] * df["Price"]
df = df[~df["Invoice"].astype(str).str.startswith("C")]

print(f"  Rows      : {len(df):,}")
print(f"  Customers : {df['Customer ID'].nunique():,}")
print(f"  Invoices  : {df['Invoice'].nunique():,}")
print(
    f"  Date range: {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}"
)
print(f"  Revenue   : £{df['revenue'].sum():,.0f}")

# ── 3. RFM feature computation ────────────────────────────────────────────────
print("\nComputing RFM features...")
snapshot = df["InvoiceDate"].max() + timedelta(weeks=1)

rfm = (
    df.groupby("Customer ID")
    .agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        total_items=("Quantity", "sum"),
        countries=("Country", "nunique"),
    )
    .reset_index()
)
rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
rfm = rfm.drop(columns=["last_purchase"])

# ── 4. RFM scoring (1–5 quintiles) ───────────────────────────────────────────
rfm["R"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(
    rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
).astype(int)
rfm["M"] = pd.qcut(
    rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["rfm_score"] = rfm["R"] * 100 + rfm["F"] * 10 + rfm["M"]
rfm["rfm_total"] = rfm["R"] + rfm["F"] + rfm["M"]


# ── 5. Customer segmentation via RFM rules ───────────────────────────────────
def segment(row):
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal customers"
    elif r >= 4 and f <= 2:
        return "New customers"
    elif r >= 3 and f >= 2 and m >= 3:
        return "Potential loyalists"
    elif r == 2 and f >= 2:
        return "At risk"
    elif r <= 2 and f >= 3:
        return "Cannot lose them"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    else:
        return "Needs attention"


rfm["segment"] = rfm.apply(segment, axis=1)

seg_counts = rfm["segment"].value_counts()
print("  Segments:")
for seg, count in seg_counts.items():
    print(f"    {seg:<22} {count:>5} customers ({count/len(rfm)*100:.1f}%)")

# ── 6. Customer Lifetime Value — BG/NBD model ─────────────────────────────────
print("\nCalculating Customer Lifetime Value...")

customer_dates = df.groupby("Customer ID")["InvoiceDate"].agg(["min", "max"])
customer_dates["lifespan_weeks"] = (
    (customer_dates["max"] - customer_dates["min"]).dt.days / 7
).clip(lower=1)
rfm = rfm.merge(customer_dates[["lifespan_weeks"]], on="Customer ID", how="left")
rfm["weekly_purchase_rate"] = rfm["frequency"] / rfm["lifespan_weeks"]
rfm["clv_historical"] = rfm["monetary"]

CLV_MODEL_TYPE = "linear_fallback"

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data

    print("  Fitting BG/NBD model (lifetimes library)...")
    # Build frequency/recency/T summary for lifetimes
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="revenue",
        observation_period_end=snapshot,
        freq="W",
    )
    # Filter to customers with at least 1 repeat purchase (BG/NBD requirement)
    summary_fit = summary[summary["frequency"] > 0].copy()

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_fit["frequency"], summary_fit["recency"], summary_fit["T"])

    # Predict expected number of purchases in next 52 weeks
    summary_fit["predicted_purchases"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            52, summary_fit["frequency"], summary_fit["recency"], summary_fit["T"]
        )
    )

    # Gamma-Gamma spend model (requires frequency > 0 and monetary > 0)
    ggf_data = summary_fit[summary_fit["monetary_value"] > 0].copy()
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(ggf_data["frequency"], ggf_data["monetary_value"])

    # Customer-level CLV = predicted_purchases × expected_avg_order_value / (1 + discount)
    clv_bgnbd = ggf.customer_lifetime_value(
        bgf,
        ggf_data["frequency"],
        ggf_data["recency"],
        ggf_data["T"],
        ggf_data["monetary_value"],
        time=12,  # months
        discount_rate=DISCOUNT_RATE / 12,  # monthly discount
        freq="W",
    ).rename("clv_projected_12m")

    # Merge back — customers with only 1 purchase get linear fallback
    rfm = rfm.merge(clv_bgnbd.reset_index(), on="Customer ID", how="left")
    # Fallback for single-purchase customers
    fallback = (
        rfm["weekly_purchase_rate"] * rfm["avg_order_value"] * 52 / (1 + DISCOUNT_RATE)
    )
    rfm["clv_projected_12m"] = rfm["clv_projected_12m"].fillna(fallback).round(2)

    CLV_MODEL_TYPE = "BG/NBD + Gamma-Gamma"
    print(f"  CLV model: {CLV_MODEL_TYPE}")
    print(f"  Customers modelled: {len(summary_fit):,} (frequency > 0)")

except ImportError:
    print("  lifetimes not installed — using enhanced linear CLV projection.")
    print("  Install with: pip install lifetimes")
    PROJECTION_WEEKS = 52
    rfm["clv_projected_12m"] = (
        rfm["weekly_purchase_rate"]
        * rfm["avg_order_value"]
        * PROJECTION_WEEKS
        / (1 + DISCOUNT_RATE)
    ).round(2)
    CLV_MODEL_TYPE = "linear_projection"

rfm["clv_model_type"] = CLV_MODEL_TYPE
print(f"  Avg CLV (projected 12m): £{rfm['clv_projected_12m'].mean():,.2f}")
print(f"  Median CLV             : £{rfm['clv_projected_12m'].median():,.2f}")

# ── 7. Churn risk — Logistic Regression (replaces heuristic) ─────────────────
print("\nModelling churn risk with Logistic Regression...")

# Define churn: customer not seen in last N days (relative to snapshot)
rfm["churned"] = (rfm["recency"] > CHURN_THRESHOLD_DAYS).astype(int)

# R/F/M removed: they are quintile-binned versions of recency/frequency/monetary,
# so including both would be redundant and inflate feature importance misleadingly.
churn_features = ["recency", "frequency", "monetary", "avg_order_value"]
X_churn = rfm[churn_features].fillna(0)
y_churn = rfm["churned"]

scaler_churn = StandardScaler()
X_churn_scaled = scaler_churn.fit_transform(X_churn)

lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)

# Cross-validated predicted probabilities (5-fold)
churn_proba = cross_val_predict(
    lr, X_churn_scaled, y_churn, cv=5, method="predict_proba"
)[:, 1]
rfm["churn_risk_score"] = churn_proba.round(3)

# Fit on all data for final model
lr.fit(X_churn_scaled, y_churn)

# Evaluate
auc = roc_auc_score(y_churn, churn_proba)
print(f"  Churn model AUC (5-fold CV): {auc:.3f}")
print(f"  Churn rate in dataset: {y_churn.mean()*100:.1f}%")


def churn_tier(score):
    if score < 0.25:
        return "Low"
    elif score < 0.5:
        return "Medium"
    elif score < 0.75:
        return "High"
    else:
        return "Critical"


rfm["churn_risk"] = rfm["churn_risk_score"].apply(churn_tier)

churn_counts = rfm["churn_risk"].value_counts()
print("  Churn risk distribution:")
for tier, count in churn_counts.items():
    print(f"    {tier:<10} {count:>5} customers ({count/len(rfm)*100:.1f}%)")

# ── 8. K-Means clustering — elbow + silhouette validation ────────────────────
print("\nFitting K-Means clusters with elbow method validation...")

cluster_features = ["recency", "frequency", "monetary"]
X = rfm[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test k=2..8 and log inertia + silhouette
diag_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(
        X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=42
    )
    diag_rows.append({"k": k, "inertia": km.inertia_, "silhouette": round(sil, 4)})

diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(OUT / "kmeans_diagnostics.csv", index=False)
print("  K-Means diagnostics (inertia + silhouette):")
print(diag_df.to_string(index=False))

# k=4 selection rationale:
# The silhouette score peaks at k=2 (0.92), which separates active vs inactive
# customers — correct statistically but not actionable for CRM targeting.
# k=4 (silhouette=0.59) maps directly to standard marketing tiers: High value,
# Mid value, Low value, and Inactive — four segments that drive distinct retention
# and upsell strategies. In practice, business segmentation need takes precedence
# over the mathematical optimum when the simpler solution lacks actionability.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_scaled)

cluster_labels = (
    rfm.groupby("cluster")["monetary"]
    .mean()
    .rank(ascending=False)
    .astype(int)
    .map({1: "High value", 2: "Mid value", 3: "Low value", 4: "Inactive"})
)
rfm["cluster_label"] = rfm["cluster"].map(cluster_labels)

# ── 9. Save outputs ───────────────────────────────────────────────────────────
print("\nSaving outputs...")

rfm_cols = [
    "Customer ID",
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_items",
    "R",
    "F",
    "M",
    "rfm_score",
    "rfm_total",
    "segment",
]
rfm[rfm_cols].to_csv(OUT / "rfm_segments.csv", index=False)
print(f"  rfm_segments.csv    : {len(rfm):,} customers")

clv_cols = [
    "Customer ID",
    "clv_historical",
    "clv_projected_12m",
    "clv_model_type",
    "weekly_purchase_rate",
    "lifespan_weeks",
    "churned",
    "churn_risk_score",
    "churn_risk",
    "cluster",
    "cluster_label",
]
rfm[clv_cols].to_csv(OUT / "clv_scores.csv", index=False)
print(f"  clv_scores.csv      : {len(rfm):,} customers")

summary_cols = [
    "Customer ID",
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "R",
    "F",
    "M",
    "rfm_score",
    "segment",
    "clv_historical",
    "clv_projected_12m",
    "clv_model_type",
    "churned",
    "churn_risk_score",
    "churn_risk",
    "cluster_label",
]
rfm[summary_cols].to_csv(OUT / "customer_summary.csv", index=False)
print(f"  customer_summary.csv: {len(rfm):,} customers")
print(f"  kmeans_diagnostics.csv")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 4 complete — customer_intelligence.py")
print(f"  Total customers     : {len(rfm):,}")
print(f"  CLV model           : {CLV_MODEL_TYPE}")
print(f"  Avg CLV (projected) : £{rfm['clv_projected_12m'].mean():,.2f}")
print(f"  Median CLV          : £{rfm['clv_projected_12m'].median():,.2f}")
print(f"  Churn model AUC     : {auc:.3f}")
print(f"  Top segment         : {seg_counts.index[0]} ({seg_counts.iloc[0]} customers)")
print(
    f"  High/Critical churn : {(rfm['churn_risk'].isin(['High','Critical'])).sum():,} customers"
)
print(
    f"  Champions revenue   : £{rfm[rfm['segment']=='Champions']['monetary'].sum():,.0f}"
)
print("=" * 55)
print("Next step -> python generate_dashboard.py")
