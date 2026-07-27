"""
feature_engineering.py  —  Step 2: Weekly Feature Engineering

FORECASTING IMPROVEMENTS (addressing examiner feedback):
The examiner's two critiques were:
  (a) Model does not use previous days/periods properly
  (b) Impact of weekends and festivals is not shown

Root causes identified and fixed:

1. SNAP — integer count (0-7) replaces binary flag.
   M5 calendar shows 0-7 SNAP days per week; a count carries
   far more signal than a 0/1 flag.

2. Event features enriched:
   a. n_events_in_week    — 0, 1 or 2 events in same week
   b. has_major_holiday   — SuperBowl/Thanksgiving/Christmas/Easter/etc
   c. event_week_before   — major holiday next week (pre-event surge)
   d. event_week_after    — major holiday last week (post-event hangover)
   e. days_since_last_event — continuous fade from last event
   f. is_thanksgiving, is_christmas, is_superbowl, is_easter,
      is_independence, is_laborday, is_mothersday — named flags
      because each event has a different demand signature

3. is_month_end — 1 if this week is the last week of the calendar month.
   Captures end-of-month spending cycles (salary payments, promotional resets).
   Replaces weekend_frac, which was constant (2/7) for every ISO week and had SHAP=0.

4. lag_3 added — fills the gap between lag_2 and lag_4, captures
   monthly cycles more precisely.

5. yoy_growth — rolling 4-week mean / same 4 weeks 52 weeks ago.
   Captures long-run growth trajectory per series.

6. price_vs_median_ratio — relative price vs 8-week own median,
   capturing price elasticity.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("processed_data")
RAW = Path("raw_data/m5")
TOP_N = 200
MIN_WEEKS = 52

VALID_STORES = {
    "CA_1",
    "CA_2",
    "CA_3",
    "CA_4",
    "TX_1",
    "TX_2",
    "TX_3",
    "WI_1",
    "WI_2",
    "WI_3",
}

MAJOR_HOLIDAYS = {
    "SuperBowl",
    "Thanksgiving",
    "Christmas",
    "OrthodoxChristmas",
    "IndependenceDay",
    "LaborDay",
    "Easter",
    "OrthodoxEaster",
    "Mother's day",
    "Father's day",
    "Halloween",
    "NewYear",
    "ChristmasEve",
    "Chanukah End",
    "Purim End",
    "MartinLutherKingDay",
    "PresidentsDay",
    "MemorialDay",
    "ColumbusDay",
    "VeteransDay",
}

STATE_MAP = {
    "CA_1": "CA",
    "CA_2": "CA",
    "CA_3": "CA",
    "CA_4": "CA",
    "TX_1": "TX",
    "TX_2": "TX",
    "TX_3": "TX",
    "WI_1": "WI",
    "WI_2": "WI",
    "WI_3": "WI",
}

print("=" * 60)
print("Step 2: Feature Engineering")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\nLoading orders, products, calendar...")
orders = pd.read_csv(OUT / "orders.csv", low_memory=False)
orders = orders[orders["order_id"] != "order_id"].copy()
orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
orders["total_amount"] = pd.to_numeric(orders["total_amount"], errors="coerce")
orders = orders.dropna(subset=["order_date"])
orders = orders[orders["store_id"].isin(VALID_STORES)]

products = pd.read_csv(OUT / "products.csv")
calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
calendar["week"] = calendar["date"].dt.to_period("W").apply(lambda r: r.start_time)

print(f"  Orders: {len(orders):,} | Calendar days: {len(calendar)}")

# ── 2. Build enriched weekly calendar ─────────────────────────────────────────
print("Building enriched weekly calendar...")

# SNAP day counts per week per state (integer 0-7)
snap_weekly = (
    calendar.groupby("week")
    .agg(
        snap_days_CA=("snap_CA", "sum"),
        snap_days_TX=("snap_TX", "sum"),
        snap_days_WI=("snap_WI", "sum"),
    )
    .reset_index()
)

# Event features per week
rows = []
for week, grp in calendar.groupby("week"):
    events1 = grp["event_name_1"].dropna().tolist()
    events2 = grp["event_name_2"].dropna().tolist()
    all_ev = events1 + events2
    types1 = grp["event_type_1"].dropna().tolist()
    types2 = grp["event_type_2"].dropna().tolist()
    all_ty = types1 + types2
    rows.append(
        {
            "week": week,
            "has_event": int(len(all_ev) > 0),
            "n_events": len(all_ev),
            "has_major_holiday": int(any(e in MAJOR_HOLIDAYS for e in all_ev)),
            "is_sporting": int("Sporting" in all_ty),
            "is_cultural": int("Cultural" in all_ty),
            "is_national": int("National" in all_ty),
            "is_religious": int("Religious" in all_ty),
            "is_thanksgiving": int(any("Thanksgiving" in e for e in all_ev)),
            "is_christmas": int(any("Christmas" in e for e in all_ev)),
            "is_superbowl": int(any("SuperBowl" in e for e in all_ev)),
            "is_easter": int(any("Easter" in e for e in all_ev)),
            "is_independence": int(any("IndependenceDay" in e for e in all_ev)),
            "is_laborday": int(any("LaborDay" in e for e in all_ev)),
            "is_mothersday": int(any("Mother" in e for e in all_ev)),
            # is_month_end: 1 if this week is the last week of its calendar month.
            # Captures end-of-month spending cycles (salary payments, budget flushes).
            # Replaces weekend_frac which was constant (2/7) for every ISO week.
            "is_month_end": int(
                (week + pd.Timedelta(weeks=1)).month != week.month
            ),
        }
    )

cal_weekly = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
cal_weekly = cal_weekly.merge(snap_weekly, on="week", how="left")

# Pre-event (next week has major holiday) and post-event (last week had one)
cal_weekly["event_week_before"] = (
    cal_weekly["has_major_holiday"].shift(-1).fillna(0).astype(int)
)
cal_weekly["event_week_after"] = (
    cal_weekly["has_major_holiday"].shift(1).fillna(0).astype(int)
)

# Days since last event (continuous proximity signal)
last_idx = -1
days_list = []
for i, r in cal_weekly.iterrows():
    if r["has_event"] == 1:
        last_idx = i
    days_list.append((i - last_idx) * 7 if last_idx >= 0 else 365)
cal_weekly["days_since_last_event"] = days_list

print(
    f"  Event weeks: {cal_weekly.has_event.sum()} | Major: {cal_weekly.has_major_holiday.sum()}"
)
print(f"  Max SNAP days/week (CA): {cal_weekly.snap_days_CA.max()}")

# ── 3. Top products ───────────────────────────────────────────────────────────
print("\nFinding top products (chunked)...")
CHUNK = 2_000_000
rev_acc = {}
for chunk in pd.read_csv(OUT / "line_items.csv", low_memory=False, chunksize=CHUNK):
    chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce")
    chunk["unit_price"] = pd.to_numeric(chunk["unit_price"], errors="coerce")
    chunk = chunk.dropna(subset=["quantity", "unit_price"])
    chunk["revenue"] = chunk["quantity"] * chunk["unit_price"]
    for pid, rev in chunk.groupby("product_id")["revenue"].sum().items():
        rev_acc[pid] = rev_acc.get(pid, 0) + rev

prod_rev = pd.Series(rev_acc).sort_values(ascending=False)
top_products = prod_rev.head(TOP_N).index.tolist()
top_set = set(top_products)
pd.DataFrame({"product_id": top_products}).to_csv(OUT / "top_products.csv", index=False)
print(
    f"  Top {TOP_N} products cover {prod_rev[top_products].sum()/prod_rev.sum():.1%} of revenue"
)

# ── 4. Aggregate to weekly grain ──────────────────────────────────────────────
print("\nAggregating to weekly grain (chunked)...")
order_lookup = orders.set_index("order_id")[["order_date", "store_id"]]
weekly_acc = []

for chunk in pd.read_csv(OUT / "line_items.csv", low_memory=False, chunksize=CHUNK):
    chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce")
    chunk["unit_price"] = pd.to_numeric(chunk["unit_price"], errors="coerce")
    chunk = chunk.dropna(subset=["quantity", "unit_price"])
    chunk = chunk[chunk["product_id"].isin(top_set)]
    if len(chunk) == 0:
        continue
    chunk = chunk.join(order_lookup, on="order_id", how="left")
    chunk["order_date"] = pd.to_datetime(chunk["order_date"])
    chunk["revenue"] = chunk["quantity"] * chunk["unit_price"]
    chunk["week"] = chunk["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
    agg = (
        chunk.groupby(["product_id", "store_id", "week"])
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            unit_price=("unit_price", "mean"),
        )
        .reset_index()
    )
    weekly_acc.append(agg)

print("  Combining...")
weekly = (
    pd.concat(weekly_acc, ignore_index=True)
    .groupby(["product_id", "store_id", "week"])
    .agg(
        quantity=("quantity", "sum"),
        revenue=("revenue", "sum"),
        unit_price=("unit_price", "mean"),
    )
    .reset_index()
)

# ── 5. Zero-fill ──────────────────────────────────────────────────────────────
print("Filling zero-sales weeks...")
all_weeks = pd.period_range(
    weekly["week"].min(), weekly["week"].max(), freq="W"
).to_timestamp()
idx = pd.MultiIndex.from_product(
    [top_products, list(VALID_STORES), all_weeks],
    names=["product_id", "store_id", "week"],
)
weekly = (
    weekly.set_index(["product_id", "store_id", "week"])
    .reindex(idx, fill_value=0)
    .reset_index()
).sort_values(["product_id", "store_id", "week"])

weekly["unit_price"] = weekly.groupby(["product_id", "store_id"])[
    "unit_price"
].transform(lambda x: x.replace(0, np.nan).ffill().bfill())
weekly["unit_price"] = weekly["unit_price"].fillna(
    weekly.groupby("product_id")["unit_price"].transform("median")
)

# ── 6. Merge enriched calendar ────────────────────────────────────────────────
print("Merging enriched calendar features...")
weekly = weekly.merge(cal_weekly, on="week", how="left")
weekly["state"] = weekly["store_id"].map(STATE_MAP)
weekly["snap_days"] = np.where(
    weekly["state"] == "CA",
    weekly["snap_days_CA"],
    np.where(weekly["state"] == "TX", weekly["snap_days_TX"], weekly["snap_days_WI"]),
)
weekly["snap"] = (weekly["snap_days"] > 0).astype(int)

# ── 7. Feature engineering ────────────────────────────────────────────────────
print("Engineering features...")
df = weekly.copy()
grp = df.groupby(["product_id", "store_id"])["quantity"]

# Lag features — lag_3 added to bridge the gap between lag_2 and lag_4
for lag in [1, 2, 3, 4, 8, 12, 52]:
    df[f"lag_{lag}"] = grp.shift(lag)

# Rolling stats
for window in [4, 8, 12]:
    df[f"rolling_mean_{window}"] = grp.shift(1).transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f"rolling_std_{window}"] = grp.shift(1).transform(
        lambda x: x.rolling(window, min_periods=2).std().fillna(0)
    )

# Year-over-year growth
roll4 = grp.shift(1).transform(lambda x: x.rolling(4, min_periods=1).mean())
roll4_yago = grp.shift(53).transform(lambda x: x.rolling(4, min_periods=1).mean())
df["yoy_growth"] = (roll4 / (roll4_yago + 1e-6)).clip(0, 5)

# Price features
pg = df.groupby(["product_id", "store_id"])["unit_price"]
df["price_lag_1"] = pg.shift(1)
df["price_change"] = (df["unit_price"] - df["price_lag_1"]) / (df["price_lag_1"] + 1e-6)
roll_med8 = pg.shift(1).transform(lambda x: x.rolling(8, min_periods=1).median())
df["price_vs_median_ratio"] = (df["unit_price"] / (roll_med8 + 1e-6)).clip(0, 5)

# Calendar
df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
df["month"] = df["week"].dt.month.astype(int)
df["quarter"] = df["week"].dt.quarter.astype(int)
df["year"] = df["week"].dt.year.astype(int)
df["is_q4"] = df["month"].isin([10, 11, 12]).astype(int)
df["is_january"] = (df["month"] == 1).astype(int)
df["weeks_since_start"] = df.groupby(["product_id", "store_id"]).cumcount()

assert df["is_q4"].sum() > 0, "is_q4 all zeros — calendar bug"

# Encoded IDs
df["store_encoded"] = df["store_id"].map(
    {s: i for i, s in enumerate(sorted(VALID_STORES))}
)
df["product_encoded"] = df["product_id"].map(
    {p: i for i, p in enumerate(sorted(top_products))}
)
df = df.merge(
    products[["product_id", "dept_id", "cat_id"]], on="product_id", how="left"
)
df["cat_encoded"] = df["cat_id"].map(
    {c: i for i, c in enumerate(sorted(df["cat_id"].dropna().unique()))}
)
df["dept_encoded"] = df["dept_id"].map(
    {d: i for i, d in enumerate(sorted(df["dept_id"].dropna().unique()))}
)

# ── 8. Clean ──────────────────────────────────────────────────────────────────
for col in ["lag_8", "lag_12", "lag_52"]:
    df[col] = df[col].fillna(0)
df["yoy_growth"] = df["yoy_growth"].fillna(1.0)
df["days_since_last_event"] = df["days_since_last_event"].fillna(365)

df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4"])

counts = df.groupby(["product_id", "store_id"])["week"].count()
valid = counts[counts >= MIN_WEEKS].index
df = df[df.set_index(["product_id", "store_id"]).index.isin(valid)].reset_index(
    drop=True
)

print(
    f"  Final rows: {len(df):,} | Series: {df.groupby(['product_id','store_id']).ngroups:,}"
)
print(f"  Nulls: {df.isnull().sum().sum()}")
print(f"  snap_days mean: {df['snap_days'].mean():.2f} | max: {df['snap_days'].max()}")
print(f"  event_week_before=1: {df['event_week_before'].sum():,}")
print(f"  yoy_growth mean: {df['yoy_growth'].mean():.3f}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUT / "weekly_features.csv", index=False)

recent = df[df["week"] >= df["week"].max() - pd.Timedelta(weeks=8)]
inv_base = (
    recent.groupby(["product_id", "store_id"])["quantity"]
    .mean()
    .reset_index()
    .rename(columns={"quantity": "avg_weekly_qty"})
)
inv_base = inv_base.merge(
    products[["product_id", "cat_id", "dept_id"]], on="product_id", how="left"
)
inv_base.to_csv(OUT / "inventory_base.csv", index=False)

print(f"\nSaved: weekly_features.csv | inventory_base.csv")
print("\n" + "=" * 60)
print("NEW features added:")
print("  snap_days     — integer count (was binary)")
print("  n_events      — number of events in week")
print("  has_major_holiday / event_week_before / event_week_after")
print("  days_since_last_event — continuous proximity signal")
print("  is_thanksgiving / is_christmas / is_superbowl / is_easter")
print("  is_independence / is_laborday / is_mothersday")
print("  lag_3         — fills monthly cycle gap")
print("  yoy_growth    — year-over-year rolling growth rate")
print("  price_vs_median_ratio — relative price elasticity")
print("  is_month_end  — last week of calendar month (replaces weekend_frac)")
print("=" * 60)
print("Next step -> python model.py")
