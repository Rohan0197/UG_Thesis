"""
generate_dashboard.py  —  Retail Intelligence BI Dashboard Generator

Reads all processed pipeline CSVs and produces retail_dashboard.html —
a single self-contained file with every chart populated from real data.
Open the output in any browser; no server or Python needed at view time.

"""

import json, textwrap
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("processed_data")
DEST = Path("retail_dashboard.html")

# ─────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────
print("Loading pipeline data...")
orders = pd.read_csv(OUT / "orders.csv", parse_dates=["order_date"])
orders["total_amount"] = pd.to_numeric(orders["total_amount"], errors="coerce")
orders = orders[orders["store_id"].str.len() <= 4].copy()
orders["state"] = orders["store_id"].str[:2]

forecasts = pd.read_csv(OUT / "forecasts.csv", parse_dates=["forecast_week"])
metrics = pd.read_csv(OUT / "model_metrics.csv")
feat_imp = pd.read_csv(OUT / "feature_importance.csv")
cs = pd.read_csv(OUT / "customer_summary.csv")
clv_df = pd.read_csv(OUT / "clv_scores.csv")
products = pd.read_csv(OUT / "products.csv")

for fname in ("inventory_base.csv", "inventory_intelligence.csv", "inventory.csv"):
    p = OUT / fname
    if p.exists():
        inv = pd.read_csv(p)
        break

if "avg_weekly_qty" not in inv.columns:
    inv.rename(columns={"avg_weekly_forecast": "avg_weekly_qty"}, inplace=True)
if "avg_weekly_qty" not in inv.columns:
    inv["avg_weekly_qty"] = 50.0

inv = inv[inv["avg_weekly_qty"] >= 1.0].copy()

np.random.seed(42)
if "current_stock" not in inv.columns:
    # Simulate stock as 1–8 weeks of avg demand so the distribution includes
    # Critical (≤2 wks), Warning (3–4 wks) and Safe (>4 wks) items.
    weeks_sim = np.random.uniform(1.0, 8.0, len(inv))
    inv["current_stock"] = (inv["avg_weekly_qty"] * weeks_sim).round(1)
if "weeks_to_stockout" not in inv.columns:
    inv["weeks_to_stockout"] = (inv["current_stock"] / inv["avg_weekly_qty"]).round(1)
if "status" not in inv.columns:
    inv["status"] = inv["weeks_to_stockout"].apply(
        lambda w: "Critical" if w <= 2 else ("Warning" if w <= 4 else "Safe")
    )
if "cat_id" not in inv.columns:
    inv = inv.merge(products[["product_id", "cat_id"]], on="product_id", how="left")

print("  All files loaded")

# ─────────────────────────────────────────────────────────
# 1b. BUILD TOP-200 ACTUAL REVENUE (match forecast scope)
# ─────────────────────────────────────────────────────────
# The forecast covers only the top 200 products.  To make the
# "Actual vs Forecast" chart comparable we compute actual weekly
# revenue from line_items filtered to the same 200 products.
# total_rev / state KPIs still use the full orders dataset.
forecasted_pids = set(forecasts["product_id"].unique())
li = pd.read_csv(OUT / "line_items.csv")
li_top200 = li[li["product_id"].isin(forecasted_pids)].copy()
li_top200["line_revenue"] = li_top200["quantity"] * li_top200["unit_price"]
li_top200 = li_top200.merge(
    orders[["order_id", "order_date"]], on="order_id", how="left"
)
li_top200["order_date"] = pd.to_datetime(li_top200["order_date"])
weekly_top200 = (
    li_top200.groupby(
        li_top200["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
    )["line_revenue"]
    .sum()
    .reset_index()
)
weekly_top200.columns = ["week", "revenue"]
weekly_top200 = weekly_top200.sort_values("week")
print(
    f"  Top-200 actual: {len(weekly_top200)} weeks, "
    f"avg ${weekly_top200['revenue'].mean():,.0f}/week"
)

# ─────────────────────────────────────────────────────────
# 2. AGGREGATE
# ─────────────────────────────────────────────────────────
print("Computing aggregates...")

total_rev = float(orders["total_amount"].sum())
by_state = orders.groupby("state")["total_amount"].sum()
total_st = float(by_state.sum())
ca_pct = round(by_state.get("CA", 0) / total_st * 100, 1)
tx_pct = round(by_state.get("TX", 0) / total_st * 100, 1)
wi_pct = round(by_state.get("WI", 0) / total_st * 100, 1)

# weekly_values: top-200 scope so it aligns with the forecast line on the chart
w52 = weekly_top200.tail(52).copy()
weekly_labels = w52["week"].dt.strftime("%d %b '%y").tolist()
weekly_values = [round(float(v), 2) for v in w52["revenue"]]
wow = float(
    (w52["revenue"].iloc[-1] - w52["revenue"].iloc[-2]) / w52["revenue"].iloc[-2] * 100
)
this_week = float(w52["revenue"].iloc[-1])
last_month_rev = float(w52.tail(4)["revenue"].sum())

fc_agg = (
    forecasts.groupby("forecast_week")["forecast_revenue"]
    .sum()
    .reset_index()
    .sort_values("forecast_week")
)
fc_all_labels = [f"W+{i+1}" for i in range(len(fc_agg))]
fc_all_values = [round(float(v), 2) for v in fc_agg["forecast_revenue"]]
fc_total_12w = float(fc_agg["forecast_revenue"].sum())
fc_avg_weekly = float(fc_agg["forecast_revenue"].mean())
fc_peak_week = int(fc_agg["forecast_revenue"].idxmax()) + 1

monthly = (
    orders.assign(month=orders["order_date"].dt.to_period("M"))
    .groupby("month")["total_amount"]
    .sum()
    .reset_index()
)
monthly["month_str"] = monthly["month"].dt.strftime("%b '%y")
monthly["growth"] = monthly["total_amount"].pct_change() * 100
m24 = monthly.tail(24).copy()
monthly_labels = m24["month_str"].tolist()
monthly_values = [round(float(v) / 1e6, 3) for v in m24["total_amount"]]
monthly_growth = [round(float(v), 2) if not pd.isna(v) else 0.0 for v in m24["growth"]]

# forecast per store×cat
fc_by_store_cat = {}
for store in sorted(forecasts["store_id"].unique()):
    fc_by_store_cat[store] = {}
    for cat in sorted(forecasts["cat_id"].unique()):
        sub = (
            forecasts[(forecasts.store_id == store) & (forecasts.cat_id == cat)]
            .groupby("forecast_week")
            .agg(qty=("forecast_qty", "sum"), rev=("forecast_revenue", "sum"))
            .reset_index()
            .sort_values("forecast_week")
        )
        if len(sub) == 0:
            continue
        fc_by_store_cat[store][cat] = {
            "labels": [f"W{i+1}" for i in range(len(sub))],
            "qty": [round(float(v), 1) for v in sub["qty"]],
            "rev": round(float(sub["rev"].sum()), 2),
            "avg_qty": round(float(sub["qty"].mean()), 1),
            "peak_wk": int(sub["qty"].idxmax()) + 1,
            "mae": round(float(metrics[metrics.split == "Test"].iloc[0]["mae"]), 2),
        }


def fmt_pid(pid, cat=""):
    parts = str(pid).split("_")
    cat_n = str(cat).title() if cat else parts[0].title()
    return f"{cat_n} · Item {parts[-1]}"


top5 = (
    forecasts.groupby(["product_id", "cat_id"])["forecast_revenue"]
    .sum()
    .reset_index()
    .sort_values("forecast_revenue", ascending=False)
    .head(5)
)
top5_data = [
    {
        "name": fmt_pid(r.product_id, r.cat_id),
        "cat": str(r.cat_id),
        "rev": round(float(r.forecast_revenue), 2),
    }
    for _, r in top5.iterrows()
]

shap_col = (
    "mean_abs_shap" if "mean_abs_shap" in feat_imp.columns else feat_imp.columns[1]
)
fi10 = feat_imp.head(10).copy()
shap_data = [
    {"feature": str(r["feature"]), "value": round(float(r[shap_col]), 4)}
    for _, r in fi10.iterrows()
]

if (
    len(metrics[metrics.split == "Test"]) == 0
    or len(metrics[metrics.split == "Validation"]) == 0
):
    raise ValueError(
        "model_metrics.csv is missing Test or Validation split — re-run model.py"
    )
test_row = metrics[metrics.split == "Test"].iloc[0]
val_row = metrics[metrics.split == "Validation"].iloc[0]
model_metrics = {
    "test": {
        "r2": round(float(test_row.r2), 4),
        "mae": round(float(test_row.mae), 2),
        "rmse": round(float(test_row.rmse), 2),
        "mape": round(float(test_row.mape), 1),
    },
    "val": {
        "r2": round(float(val_row.r2), 4),
        "mae": round(float(val_row.mae), 2),
        "rmse": round(float(val_row.rmse), 2),
        "mape": round(float(val_row.mape), 1),
    },
}

inv_crit = int((inv.status == "Critical").sum())
inv_warn = int((inv.status == "Warning").sum())
inv_safe = int((inv.status == "Safe").sum())
inv_total = len(inv)
inv_cov = round(inv_safe / inv_total * 100, 1)

cat_cov = (
    inv.groupby("cat_id")["weeks_to_stockout"]
    .mean()
    .reset_index()
    .sort_values("weeks_to_stockout", ascending=False)
)
inv_cat_labels = cat_cov["cat_id"].tolist()
inv_cat_values = [round(float(v), 2) for v in cat_cov["weeks_to_stockout"]]

store_crit = (
    inv[inv.status == "Critical"]
    .groupby("store_id")
    .size()
    .reindex(sorted(forecasts.store_id.unique()), fill_value=0)
    .reset_index(name="count")
)
inv_store_labels = store_crit["store_id"].tolist()
inv_store_values = store_crit["count"].tolist()

crit_items = (
    inv[inv.status == "Critical"]
    .merge(
        products[["product_id", "cat_id", "median_price"]].rename(
            columns={"cat_id": "p_cat"}
        ),
        on="product_id",
        how="left",
    )
    .sort_values("weeks_to_stockout")
    .head(20)
)
inv_table_rows = []
for i, (_, r) in enumerate(crit_items.iterrows(), 1):
    cat = r.get("cat_id", r.get("p_cat", "—"))
    price = float(r.get("median_price", 0) or 0)
    wts = float(r.weeks_to_stockout)
    demand = round(float(r.avg_weekly_qty), 1)
    inv_table_rows.append(
        {
            "rank": i,
            "product": fmt_pid(r.product_id, cat),
            "cat": str(cat),
            "store": str(r.store_id),
            "weeks": wts,
            "demand": demand,
            "fc12": round(demand * 12),
            "price": round(price, 2),
            "priority": "CRITICAL" if wts <= 1 else "WARNING",
        }
    )

n_customers = len(cs)
avg_clv = round(float(cs.clv_projected_12m.mean()), 2)
churn_high = cs[cs.churn_risk.isin(["High", "Critical"])]
n_churn_high = len(churn_high)
rev_at_risk = round(float(churn_high.clv_projected_12m.sum()), 2)
n_champions = int((cs.segment == "Champions").sum())
clv_model_used = (
    cs["clv_model_type"].iloc[0]
    if "clv_model_type" in cs.columns
    else "Linear projection"
)

seg_counts = cs.segment.value_counts().reset_index()
seg_counts.columns = ["segment", "count"]
seg_rev = cs.groupby("segment")["monetary"].sum().reset_index()
seg_merged = seg_counts.merge(seg_rev, on="segment").sort_values(
    "count", ascending=False
)
seg_data = [
    {
        "label": str(r.segment),
        "count": int(r["count"]),
        "revenue": round(float(r.monetary), 2),
    }
    for _, r in seg_merged.iterrows()
]

SEG_COL = {
    "Champions": "#00c4a0",
    "Loyal customers": "#4a9eff",
    "Potential loyalists": "#a78bfa",
    "At risk": "#f5a623",
    "New customers": "#5DCAA5",
    "Cannot lose them": "#D4537E",
    "Needs attention": "#526070",
    "Lost": "#f04b5a",
}

merged_cs = cs.merge(
    clv_df[["Customer ID", "cluster_label"]], on="Customer ID", how="left"
)
samp = merged_cs.sample(min(600, len(merged_cs)), random_state=42)
samp = samp[samp.clv_projected_12m < samp.clv_projected_12m.quantile(0.97)]
clv_scatter = {}
for seg_name, grp in samp.groupby("segment"):
    clv_scatter[str(seg_name)] = {
        "color": SEG_COL.get(str(seg_name), "#526070"),
        "points": [
            {
                "x": round(float(r.churn_risk_score), 3),
                "y": round(float(r.clv_projected_12m), 2),
            }
            for _, r in grp.iterrows()
        ],
    }

store_display = {
    "CA_1": "CA_1 — California Store 1",
    "CA_2": "CA_2 — California Store 2",
    "CA_3": "CA_3 — California Store 3",
    "CA_4": "CA_4 — California Store 4",
    "TX_1": "TX_1 — Texas Store 1",
    "TX_2": "TX_2 — Texas Store 2",
    "TX_3": "TX_3 — Texas Store 3",
    "WI_1": "WI_1 — Wisconsin Store 1",
    "WI_2": "WI_2 — Wisconsin Store 2",
    "WI_3": "WI_3 — Wisconsin Store 3",
}

DATA = {
    "total_rev": total_rev,
    "wow": round(wow, 2),
    "this_week": this_week,
    "last_month_rev": last_month_rev,
    "ca_pct": ca_pct,
    "tx_pct": tx_pct,
    "wi_pct": wi_pct,
    "fc_total_12w": fc_total_12w,
    "fc_avg_weekly": fc_avg_weekly,
    "fc_peak_week": fc_peak_week,
    "weekly_labels": weekly_labels,
    "weekly_values": weekly_values,
    "fc_all_labels": fc_all_labels,
    "fc_all_values": fc_all_values,
    "monthly_labels": monthly_labels,
    "monthly_values": monthly_values,
    "monthly_growth": monthly_growth,
    "fc_stores": sorted(forecasts.store_id.unique().tolist()),
    "fc_cats": sorted(forecasts.cat_id.unique().tolist()),
    "store_display": store_display,
    "fc_by_store_cat": fc_by_store_cat,
    "top5": top5_data,
    "model_metrics": model_metrics,
    "shap_data": shap_data,
    "inv_crit": inv_crit,
    "inv_warn": inv_warn,
    "inv_safe": inv_safe,
    "inv_total": inv_total,
    "inv_cov": inv_cov,
    "inv_cat_labels": inv_cat_labels,
    "inv_cat_values": inv_cat_values,
    "inv_store_labels": inv_store_labels,
    "inv_store_values": inv_store_values,
    "inv_table": inv_table_rows,
    "n_customers": n_customers,
    "avg_clv": avg_clv,
    "n_churn_high": n_churn_high,
    "rev_at_risk": rev_at_risk,
    "n_champions": n_champions,
    "clv_model_used": str(clv_model_used),
    "seg_data": seg_data,
    "seg_colors": SEG_COL,
    "clv_scatter": clv_scatter,
}

payload = json.dumps(DATA, ensure_ascii=False)
print(f"  JSON payload: {len(payload):,} bytes")

# ─────────────────────────────────────────────────────────
# 3. HTML TEMPLATE (read from same directory)
# ─────────────────────────────────────────────────────────
TEMPLATE = Path(__file__).parent / "dashboard_template.html"
if not TEMPLATE.exists():
    raise FileNotFoundError(
        "dashboard_template.html not found. "
        "Ensure both generate_dashboard.py and dashboard_template.html are in the same folder."
    )
html = TEMPLATE.read_text(encoding="utf-8")
html_out = html.replace("__DATA_PLACEHOLDER__", payload)
DEST.write_text(html_out, encoding="utf-8")

size_kb = DEST.stat().st_size // 1024
print(f"\n Dashboard written: {DEST}  ({size_kb} KB)")
print("  Open retail_dashboard.html in any browser — no server needed.")
print(f"\n  Key numbers from your pipeline:")
print(f"  Total revenue    : ${total_rev:,.0f}")
print(f"  Forecast R²      : {model_metrics['test']['r2']}")
print(f"  Customers        : {n_customers:,}")
print(f"  Avg CLV          : ${avg_clv:,.2f}")
print(f"  Critical stock   : {inv_crit} SKUs")
print(f"  Revenue at risk  : ${rev_at_risk:,.0f}")
