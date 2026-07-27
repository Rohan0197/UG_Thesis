import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("raw_data/m5")
OUT = Path("processed_data")
OUT.mkdir(exist_ok=True)

CHUNK_DAYS = 200  # days per mini-batch

# ── Load reference files (small) ──────────────────────────────────────────────
print("Loading reference files...")
calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
prices = pd.read_csv(RAW / "sell_prices.csv")

cal = calendar[
    [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
].copy()

# ── Load only the metadata + IDs from sales ───────────────────────────────────
print("Loading sales metadata...")
sales_meta = pd.read_csv(
    RAW / "sales_train_evaluation.csv",
    usecols=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
)
all_cols = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns.tolist()
day_cols = [c for c in all_cols if c.startswith("d_")]
stores = sales_meta["store_id"].unique()
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

print(f"Stores: {stores.tolist()}")
print(f"Day columns: {len(day_cols)}")
print(f"Chunk size: {CHUNK_DAYS} days\n")

# ── Products table ────────────────────────────────────────────────────────────
print("Building products table...")
product_meta = sales_meta[["item_id", "dept_id", "cat_id"]].drop_duplicates()
price_stats = (
    prices.groupby("item_id")["sell_price"]
    .agg(median_price="median", price_min="min", price_max="max")
    .reset_index()
)
products = product_meta.merge(price_stats, on="item_id", how="left")
products = products.rename(columns={"item_id": "product_id"})
products.to_csv(OUT / "products.csv", index=False)
print(f"  Products: {len(products):,}")

# ── Stores table ──────────────────────────────────────────────────────────────
print("Building stores table...")
store_meta = (
    sales_meta[["store_id", "state_id"]]
    .drop_duplicates()
    .rename(columns={"state_id": "state"})
)
store_meta["region"] = store_meta["state"].map(
    {"CA": "West", "TX": "South", "WI": "Midwest"}
)
store_meta.to_csv(OUT / "stores.csv", index=False)
print(f"  Stores: {len(store_meta)}\n")

# ── Open output files for streaming ──────────────────────────────────────────
orders_file = OUT / "orders.csv"
li_file = OUT / "line_items.csv"
orders_header = True
li_header = True

total_orders = 0
total_li = 0

# ── Main loop: store × day-chunk ──────────────────────────────────────────────
day_chunks = [day_cols[i : i + CHUNK_DAYS] for i in range(0, len(day_cols), CHUNK_DAYS)]

for store in stores:
    store_prices = prices[prices["store_id"] == store][
        ["item_id", "wm_yr_wk", "sell_price"]
    ]
    store_rows = sales_meta[sales_meta["store_id"] == store]["id"].tolist()
    print(f"Processing {store}  ({len(store_rows)} items, {len(day_chunks)} chunks)...")

    for ci, chunk in enumerate(day_chunks):
        cols_to_read = id_cols + chunk
        df_chunk = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=cols_to_read)
        df_chunk = df_chunk[df_chunk["store_id"] == store]

        long = df_chunk.melt(id_vars=id_cols, var_name="d", value_name="quantity")
        long = long[long["quantity"] > 0].copy()
        del df_chunk

        if len(long) == 0:
            continue

        long = long.merge(cal, on="d", how="left")
        long = long.merge(store_prices, on=["item_id", "wm_yr_wk"], how="left")
        long["sell_price"] = long.groupby("item_id")["sell_price"].transform(
            lambda x: x.fillna(x.median())
        )
        long["sell_price"] = long["sell_price"].fillna(long["sell_price"].median())

        long["total_line"] = long["quantity"] * long["sell_price"]
        long["order_id"] = store + "_" + long["date"].dt.strftime("%Y%m%d")

        # Orders
        ord_agg = (
            long.groupby(["order_id", "store_id", "date"])
            .agg(
                total_amount=("total_line", "sum"),
                total_units=("quantity", "sum"),
                n_items=("item_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"date": "order_date"})
        )
        ord_agg = ord_agg[
            [
                "order_id",
                "order_date",
                "store_id",
                "total_amount",
                "total_units",
                "n_items",
            ]
        ]
        ord_agg.to_csv(
            orders_file,
            mode="w" if orders_header else "a",
            header=orders_header,
            index=False,
        )
        orders_header = False
        total_orders += len(ord_agg)

        # Line items
        li = long[
            [
                "order_id",
                "item_id",
                "quantity",
                "sell_price",
                "event_name_1",
                "event_type_1",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        ].rename(columns={"item_id": "product_id", "sell_price": "unit_price"})
        li.to_csv(
            li_file, mode="w" if li_header else "a", header=li_header, index=False
        )
        li_header = False
        total_li += len(li)

        del long, ord_agg, li

    print(
        f"  Done. Running totals -> orders: {total_orders:,}  line_items: {total_li:,}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
orders = pd.read_csv(OUT / "orders.csv", usecols=["order_date", "total_amount"])
print("\n" + "=" * 50)
print("Step 1 complete — data_builder.py")
print(f"  Date range : {orders['order_date'].min()} -> {orders['order_date'].max()}")
print(f"  Orders     : {total_orders:,}")
print(f"  Line items : {total_li:,}")
print(f"  Products   : {len(products):,}")
print(f"  Stores     : {len(store_meta)}")
print(f"  Revenue    : ${float(orders['total_amount'].sum()):,.0f}")
print("=" * 50)
print("Next step -> python feature_engineering.py")
