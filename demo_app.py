"""
demo_app.py — Streamlit demo for the Multistage Recommender System.

Two-section layout:
  1. Recommended for You — personalized carousel driven by Triton on EKS
  2. Browse Catalog      — full item catalog, filterable by category;
                           clicking "Interested" feeds the behavioral feature loop

Run:
    pip install streamlit requests plotly
    streamlit run demo_app.py
"""

import datetime
import json
import os
import time
from pathlib import Path

import plotly.express as px
import requests
import streamlit as st

#config
st.set_page_config(
    page_title="Multistage Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)

LAMBDA_URL = os.environ.get(
    "LAMBDA_URL",
    "https://xxxxxxxxxxxxxxxxxxx.lambda-url.us-east-1.on.aws/",
)

DEVICE_OPTIONS = {"🖥️ Desktop": 0, "📱 Mobile": 1, "📟 Tablet": 2, "❓ Unknown": None}

TIME_PRESETS = {
    "Morning (8 am)":    8,
    "Afternoon (2 pm)": 14,
    "Evening (7 pm)":   19,
    "Night (11 pm)":    23,
    "Unknown":           None,
}

CATALOG_PATH = Path(__file__).parent / "catalog.json"
PIPELINE_WAIT_SECS = 6
CATALOG_PAGE_SIZE  = 12


# cached catalog data
@st.cache_data
def load_catalog():
    with open(CATALOG_PATH) as f:
        return json.load(f)


# Lambda helper functions
def call_lambda(payload: dict) -> dict:
    resp = requests.post(LAMBDA_URL, json=payload, timeout=35)
    resp.raise_for_status()
    return resp.json()


def fetch_recs(user_id, device_type, timestamp, top_k) -> list:
    payload = {"user_id": user_id, "top_k": top_k}
    if device_type is not None:
        payload["device_type"] = device_type
    if timestamp is not None:
        payload["timestamp"] = timestamp
    return call_lambda(payload).get("recommendations", [])


def mark_seen(user_id: int, item_ids: list):
    call_lambda({"user_id": user_id, "mark_seen": item_ids})
    st.session_state.seen_items.update(item_ids)
    if st.session_state.recs and not st.session_state.snapshot_recs:
        st.session_state.snapshot_recs = list(st.session_state.recs)
    st.session_state.pending_refresh = True


def flush_user(user_id: int):
    call_lambda({"user_id": user_id, "flush": True})
    st.session_state.seen_items     = set()
    st.session_state.snapshot_recs  = []
    st.session_state.pending_refresh = False


def make_timestamp(hour) -> int | None:
    if hour is None:
        return None
    now = datetime.datetime.now()
    return int(now.replace(hour=hour, minute=0, second=0, microsecond=0).timestamp())


# Session state
for key, default in [
    ("recs",            []),
    ("snapshot_recs",   []),
    ("seen_items",      set()),
    ("pending_refresh", False),
    ("catalog_page",    0),
    ("last_filter",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Card renderers
def rec_card(rec: dict, user_id: int, col, is_new: bool = False, key_prefix: str = "rec"):
    item_id  = rec["item_id"]
    is_seen  = item_id in st.session_state.seen_items
    score    = rec.get("score", 0)
    title    = rec.get("title", f"Item {item_id}")

    with col:
        with st.container(border=True):
            if rec.get("image_url"):
                st.image(rec["image_url"], use_container_width=True)

            if is_new:
                st.markdown(f"🆕 **{title}**")
            elif is_seen:
                st.markdown(f"~~**{title}**~~ ✅")
            else:
                st.markdown(f"**{title}**")

            st.caption(
                f"🏷️ {rec.get('category_l1','').capitalize()}"
                f"  ·  📂 {rec.get('category_l2','').capitalize()}"
            )
            st.progress(min(score, 1.0), text=f"Score: {score:.3f}")
            price = rec.get("price", "")
            if price and price not in ("None", ""):
                st.caption(f"💰 ${float(price):.2f}")

            btn_label = "✅ Seen" if is_seen else "👆 Interested"
            if st.button(btn_label, key=f"{key_prefix}_{item_id}", disabled=is_seen):
                mark_seen(user_id, [item_id])
                st.rerun()


def catalog_card(item: dict, user_id: int, col):
    item_id = item["item_id"]
    is_seen = item_id in st.session_state.seen_items

    with col:
        with st.container(border=True):
            if item.get("image_url"):
                st.image(item["image_url"], use_container_width=True)

            title = item.get("title", f"Item {item_id}")
            st.markdown(f"~~**{title}**~~ ✅" if is_seen else f"**{title}**")

            gender_label = {"M": "👔 Men", "F": "👗 Women", "U": "🧢 Unisex"}.get(
                item.get("gender", ""), ""
            )
            st.caption(
                f"🏷️ {item.get('category_l1','').capitalize()}"
                + (f"  ·  {gender_label}" if gender_label else "")
            )
            price = item.get("price", "")
            if price and price not in ("None", ""):
                st.caption(f"💰 ${float(price):.2f}")

            btn_label = "✅ Seen" if is_seen else "👆 Interested"
            if st.button(btn_label, key=f"cat_{item_id}", disabled=is_seen):
                mark_seen(user_id, [item_id])
                st.rerun()


# sidebar
with st.sidebar:
    st.markdown("## Recommender System Demo")
    st.caption("Two-Tower retrieval → DLRM ranking  \nTriton on EKS · Feast · Redis Bloom filter")
    st.divider()

    user_id      = st.number_input("👤 User ID", value=5, min_value=1, max_value=2147483647, step=1)
    device_label = st.selectbox("Device", list(DEVICE_OPTIONS.keys()))
    device_type  = DEVICE_OPTIONS[device_label]
    time_label   = st.selectbox("Time of day", list(TIME_PRESETS.keys()))
    timestamp    = make_timestamp(TIME_PRESETS[time_label])
    top_k        = st.slider("Top-K", 5, 10, value=8)

    st.divider()
    get_btn   = st.button("Get Recommendations", type="primary", use_container_width=True)
    flush_btn = st.button("🗑 Reset User State",   use_container_width=True)

    if st.session_state.seen_items:
        st.divider()
        st.markdown(f"**👁️ Interested in:** {len(st.session_state.seen_items)} item(s)")
        if st.session_state.pending_refresh:
            st.caption("Behavioral features are updating ↑")


# button actions
if flush_btn:
    with st.spinner("Resetting user state…"):
        flush_user(user_id)
    st.session_state.recs = []
    st.rerun()

if get_btn:
    with st.spinner("Querying Triton ensemble…"):
        st.session_state.recs = fetch_recs(user_id, device_type, timestamp, top_k)
    st.session_state.snapshot_recs  = []
    st.session_state.pending_refresh = False

# page header
st.title("Multistage Recommender System")
st.caption(
    "Two-Tower retrieval + DLRM ranking · Triton inference on EKS · "
    "Real-time behavioral personalization via Feast + Redis"
)

catalog = load_catalog()
all_categories = sorted(set(i["category_l1"] for i in catalog if i["category_l1"]))



# SECTION 1 — Recommended for You
# st.header("Recommended for You")
st.markdown("### *Recommended for You*")

if not st.session_state.recs:
    st.info("Select a user and click **Get Recommendations** to start.")
else:
    recs     = st.session_state.recs
    snapshot = st.session_state.snapshot_recs

    # pending refresh banner
    if st.session_state.pending_refresh:
        banner, btn_col = st.columns([3, 1])
        with banner:
            st.info(
                f"🧠 **{len(st.session_state.seen_items)} item(s)** marked as interested — "
                "behavioral features are updating via the SQS pipeline. "
                "Click Refresh to see the shift."
            )
        with btn_col:
            if st.button("⟳ Refresh Recommendations", type="primary", use_container_width=True):
                with st.spinner(f"Waiting for feature pipeline ({PIPELINE_WAIT_SECS}s)…"):
                    time.sleep(PIPELINE_WAIT_SECS)
                with st.spinner("Fetching updated recommendations…"):
                    new_recs = fetch_recs(user_id, device_type, timestamp, top_k)
                st.session_state.snapshot_recs  = list(recs)
                st.session_state.recs           = new_recs
                st.session_state.pending_refresh = False
                st.rerun()

    # before / after view
    if snapshot:
        prev_ids = {r["item_id"] for r in snapshot}
        curr_ids = {r["item_id"] for r in recs}
        new_ids  = curr_ids - prev_ids

        prev_cats = [r.get("category_l1", "") for r in snapshot]
        curr_cats = [r.get("category_l1", "") for r in recs]
        prev_top  = max(set(prev_cats), key=prev_cats.count) if prev_cats else "—"
        curr_top  = max(set(curr_cats), key=curr_cats.count) if curr_cats else "—"

        left, right = st.columns(2)
        with left:
            st.subheader("Before")
            st.caption(f"Top category: **{prev_top.capitalize()}**")
            cols = st.columns(4)
            for i, rec in enumerate(snapshot):
                rec_card(rec, user_id, cols[i % 4], key_prefix="before")
        with right:
            shift = f" → shifted to **{curr_top.capitalize()}**" if curr_top != prev_top else " (no shift yet)"
            st.subheader("After")
            st.caption(f"Top category: **{curr_top.capitalize()}**{shift}")
            cols = st.columns(4)
            for i, rec in enumerate(recs):
                rec_card(rec, user_id, cols[i % 4], is_new=(rec["item_id"] in new_ids), key_prefix="after")

    # single view with charts
    else:
        cats    = [r.get("category_l1", "") for r in recs]
        top_cat = max(set(cats), key=cats.count) if cats else "—"

        st.caption(f"User **{user_id}** · Top category: **{top_cat.capitalize()}** · Top score: **{recs[0]['score']:.3f}**")

        cols = st.columns(4)
        for i, rec in enumerate(recs):
            rec_card(rec, user_id, cols[i % 4])

        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Category distribution**")
            cat_counts: dict = {}
            for r in recs:
                c = r.get("category_l1", "?")
                cat_counts[c] = cat_counts.get(c, 0) + 1
            fig = px.bar(
                x=list(cat_counts.keys()), y=list(cat_counts.values()),
                color=list(cat_counts.keys()), height=220,
                labels={"x": "", "y": "count"},
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.caption("**Score distribution**")
            fig2 = px.bar(
                x=[r["score"] for r in recs],
                y=[r.get("title", str(r["item_id"]))[:22] for r in recs],
                orientation="h", height=220,
                labels={"x": "score", "y": ""},
            )
            fig2.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)

st.divider()



# SECTION 2 — Browse Catalog
# st.header("🛒 Browse Catalog")
st.markdown("### 🛒 Browse Catalog")
st.caption(
    "Browse items by category. Click **Interested** on anything that catches your eye — "
    "it feeds the behavioral feature loop and shifts your recommendations above."
)

f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    sel_cat    = st.selectbox("Category", ["All"] + all_categories, key="cat_filter")
with f2:
    sel_gender = st.selectbox("Gender", ["All", "M", "F", "U"], key="gender_filter")
with f3:
    sort_by    = st.selectbox("Sort by", ["Default", "Price ↑", "Price ↓"], key="sort_filter")

# filter
filtered = [
    i for i in catalog
    if (sel_cat    == "All" or i["category_l1"] == sel_cat)
    and (sel_gender == "All" or i.get("gender", "") == sel_gender)
]

# sort
if sort_by == "Price ↑":
    filtered = sorted(filtered, key=lambda x: float(x["price"] or 0))
elif sort_by == "Price ↓":
    filtered = sorted(filtered, key=lambda x: float(x["price"] or 0), reverse=True)

st.caption(f"{len(filtered)} items")

# reset page when filters change
filter_key = (sel_cat, sel_gender, sort_by)
if st.session_state.last_filter != filter_key:
    st.session_state.catalog_page = 0
    st.session_state.last_filter  = filter_key

start      = st.session_state.catalog_page * CATALOG_PAGE_SIZE
page_items = filtered[start: start + CATALOG_PAGE_SIZE]

cols = st.columns(4)
for i, item in enumerate(page_items):
    catalog_card(item, user_id, cols[i % 4])

# pagination
pg_prev, pg_label, pg_next = st.columns([1, 2, 1])
total_pages = max(1, (len(filtered) + CATALOG_PAGE_SIZE - 1) // CATALOG_PAGE_SIZE)
with pg_prev:
    if st.session_state.catalog_page > 0:
        if st.button("← Previous", use_container_width=True):
            st.session_state.catalog_page -= 1
            st.rerun()
with pg_label:
    st.caption(f"Page {st.session_state.catalog_page + 1} of {total_pages}")
with pg_next:
    if start + CATALOG_PAGE_SIZE < len(filtered):
        if st.button("Next →", use_container_width=True):
            st.session_state.catalog_page += 1
            st.rerun()