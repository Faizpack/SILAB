"""YouTube Dashboard (Streamlit version).

Run:  pip install -r requirements.txt
      streamlit run app.py

If youtube.csv sits next to this file it loads automatically,
otherwise use the uploader in the sidebar.
"""
import io
import re
from math import erf, sqrt
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="YouTube Dashboard", page_icon="📊", layout="wide")

TEAL, CORAL, GOLD = "#0f766e", "#e4572e", "#d99a1e"
MULT = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}
SUB_RE = re.compile(r"^([\d.,]+)\s*([KMB]?)\s+subscribers?$", re.I)
LINK_RE = re.compile(r"https?://|www\.")
REQUIRED = ["link", "title", "description", "category"]
DEFAULT_CSV = Path(__file__).with_name("youtube.csv")


# ---------------------------------------------------------------- helpers
def fmt(n):
    if n is None or pd.isna(n):
        return "–"
    if n >= 1e6:
        return f"{n / 1e6:.{0 if n >= 1e7 else 1}f}M"
    if n >= 1e3:
        return f"{n / 1e3:.{0 if n >= 1e4 else 1}f}K"
    return f"{round(n)}"


def pct(v):
    return "–" if v is None or pd.isna(v) else f"{v * 100:.1f}%"


def show(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def hbar(df, label, value, color=TEAL, order=None, height=None):
    if df.empty:
        st.info("No data.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(
            y=alt.Y(label, sort=order if order else "-x", title=None),
            x=alt.X(value, title=None),
            tooltip=[label, alt.Tooltip(value, format=",.3~f")],
        )
        .properties(height=height or max(120, 30 * len(df)))
    )
    show(chart)


def parse_desc(text):
    lines = [s.strip() for s in str(text).split("\n")]
    subs = np.nan
    for line in lines[1:6]:
        m = SUB_RE.match(line)
        if m:
            subs = float(m.group(1).replace(",", "")) * MULT[m.group(2).upper()]
            break
    body = re.sub(r"SHOW MORE\s*$", "", " ".join(lines[1:]))
    return (lines[0] or "Unknown"), subs, body


@st.cache_data(show_spinner="Reading file…")
def load_data(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("This file needs the columns: " + ", ".join(REQUIRED))
    df = df.dropna(subset=["link"]).drop_duplicates("link")
    parsed = df["description"].fillna("").map(parse_desc).tolist()
    return pd.DataFrame(
        {
            "cat": df["category"].fillna("").str.strip().to_numpy(),
            "channel": [p[0] for p in parsed],
            "subs": [p[1] for p in parsed],
            "tl": df["title"].fillna("").str.len().to_numpy(),
            "dw": [len(p[2].split()) for p in parsed],
            "link": [bool(LINK_RE.search(p[2])) for p in parsed],
        }
    )


# ------------------------------------------------- prediction model (ridge)
def design(df, cats, tm, ts, dm, ds):
    cols = [np.ones(len(df))]
    cols += [(df["cat"] == c).astype(float).to_numpy() for c in cats]
    cols.append(((df["tl"] - tm) / ts).to_numpy())
    cols.append(((np.log10(1 + df["dw"]) - dm) / ds).to_numpy())
    cols.append(df["link"].astype(float).to_numpy())
    return np.column_stack(cols)


@st.cache_data(show_spinner="Training model…")
def train_model(df: pd.DataFrame):
    rows = df[df["subs"] > 0].reset_index(drop=True)
    if len(rows) < 30:
        return None
    cats = sorted(df["cat"].unique())
    lw = np.log10(1 + rows["dw"])
    p = dict(
        cats=cats,
        tm=rows["tl"].mean(),
        ts=rows["tl"].std(ddof=0) or 1.0,
        dm=lw.mean(),
        ds=lw.std(ddof=0) or 1.0,
    )
    is_test = (np.arange(len(rows)) % 5) == 0
    tr, te = rows[~is_test], rows[is_test]
    Xtr = design(tr, **p)
    ytr = np.log10(tr["subs"].to_numpy())
    reg = np.eye(Xtr.shape[1])
    reg[0, 0] = 1e-6  # barely penalise the intercept
    w = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
    sigma = float(np.sqrt(np.mean((ytr - Xtr @ w) ** 2)))
    yte = np.log10(te["subs"].to_numpy())
    pred = design(te, **p) @ w
    r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    return dict(p, w=w, sigma=sigma, r2=float(r2),
                fold=float(10 ** np.mean(np.abs(yte - pred))),
                n_train=len(tr), n_test=len(te))


def predict(m, cat, tl, dw, link):
    row = pd.DataFrame({"cat": [cat], "tl": [tl], "dw": [dw], "link": [link]})
    keys = ["cats", "tm", "ts", "dm", "ds"]
    z = float((design(row, **{k: m[k] for k in keys}) @ m["w"])[0])
    p1m = 1 - 0.5 * (1 + erf(((6 - z) / m["sigma"]) / sqrt(2)))
    return z, p1m


# -------------------------------------------------------------------- UI
st.title("📊 YouTube Dashboard")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Choose youtube.csv", type="csv")
    st.caption("The file is processed locally by this app.")

raw = up.getvalue() if up else (DEFAULT_CSV.read_bytes() if DEFAULT_CSV.exists() else None)
if raw is None:
    st.info("Upload youtube.csv in the sidebar to start.")
    st.stop()

try:
    data = load_data(raw)
except Exception as e:  # noqa: BLE001
    st.error(str(e) if isinstance(e, ValueError) else "Could not read this file.")
    st.stop()

cats = sorted(data["cat"].unique())
with st.sidebar:
    st.header("Filter")
    current = st.selectbox("Category", ["All categories", *cats])

sel = data if current == "All categories" else data[data["cat"] == current]
with_s = sel.dropna(subset=["subs"])
st.caption(f"Loaded {len(data):,} unique videos.")

tab1, tab2, tab3 = st.tabs(["Overview", "Categories", "Prediction"])

# ---- Overview
with tab1:
    mega = (with_s["subs"] > 1e6).mean() if len(with_s) else np.nan
    kpis = [
        (f"{len(sel):,}", "Videos"),
        (f"{sel['channel'].nunique():,}", "Channels"),
        (fmt(with_s["subs"].median()), "Median subscribers"),
        (fmt(with_s["subs"].mean()), "Average subscribers"),
        (pct(mega), "From 1M+ channels"),
        (f"{sel['tl'].mean():.1f}", "Avg title length"),
        (f"{sel['dw'].mean():.0f}" if len(sel) else "–", "Avg description words"),
        (pct(sel["link"].mean()), "With a link"),
    ]
    for start in (0, 4):
        for col, (v, label) in zip(st.columns(4), kpis[start:start + 4]):
            col.metric(label, v)

    left, right = st.columns(2)
    with left:
        st.subheader("Subscribers spread")
        edges = [("Under 10K", 0, 1e4), ("10K–100K", 1e4, 1e5), ("100K–1M", 1e5, 1e6),
                 ("1M–10M", 1e6, 1e7), ("10M+", 1e7, np.inf)]
        hist = pd.DataFrame(
            [(n, int(((with_s["subs"] >= a) & (with_s["subs"] < b)).sum())) for n, a, b in edges],
            columns=["Size", "Videos"],
        )
        hbar(hist, "Size", "Videos", GOLD, order=[e[0] for e in edges])
    with right:
        st.subheader("Top channels")
        top = (
            sel.dropna(subset=["subs"])
            .groupby("channel", as_index=False)
            .agg(Category=("cat", "first"), Subscribers=("subs", "first"), Videos=("channel", "size"))
            .sort_values("Subscribers", ascending=False)
            .head(10)
            .rename(columns={"channel": "Channel"})
        )
        top["Subscribers"] = top["Subscribers"].map(fmt)
        st.dataframe(top, hide_index=True)

# ---- Categories
with tab2:
    g = data.groupby("cat")
    sub = data.dropna(subset=["subs"]).groupby("cat")["subs"]
    per = pd.DataFrame(
        {
            "Videos": g.size(),
            "Median subscribers": sub.median(),
            "Share from 1M+ (%)": sub.apply(lambda s: (s > 1e6).mean() * 100),
            "Avg title length": g["tl"].mean(),
            "With a link (%)": g["link"].mean() * 100,
        }
    ).reset_index().rename(columns={"cat": "Category"})
    plots = [
        ("Videos per category", "Videos", TEAL),
        ("Median subscribers", "Median subscribers", GOLD),
        ("Videos from 1M+ channels (%)", "Share from 1M+ (%)", CORAL),
        ("Average title length", "Avg title length", TEAL),
        ("Videos with a link in description (%)", "With a link (%)", CORAL),
    ]
    for (title, colname, color), col in zip(plots, [c for _ in range(3) for c in st.columns(2)]):
        with col:
            st.subheader(title)
            hbar(per[["Category", colname]].fillna(0), "Category", colname, color)

# ---- Prediction
with tab3:
    st.subheader("Prediction model: how big is the channel?")
    model = train_model(data)
    if model is None:
        st.warning("Not enough rows with subscriber counts to train a model.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        p_cat = c1.selectbox("Category", cats, key="p_cat")
        p_tl = c2.number_input("Title length (characters)", 1, 200, 60)
        p_dw = c3.number_input("Description words", 0, 2000, 80)
        p_lk = c4.selectbox("Link in description", ["Yes", "No"]) == "Yes"

        z, p1m = predict(model, p_cat, p_tl, p_dw, p_lk)
        sg = model["sigma"]
        r1, r2_, r3 = st.columns(3)
        r1.metric("Typical channel size", f"{fmt(10 ** z)} subs")
        r2_.metric("Likely range (~68%)", f"{fmt(10 ** (z - sg))} – {fmt(10 ** (z + sg))}")
        r3.metric("Chance of a 1M+ channel", pct(p1m))
        st.caption(
            f"Ridge regression on log10(subscribers). Trained on {model['n_train']:,} videos, "
            f"tested on {model['n_test']:,} unseen ones. Test R² = {model['r2']:.2f}; "
            f"a typical miss is about {model['fold']:.1f}× in subscribers. "
            "Treat it as a rough guide, not a forecast."
        )

        st.subheader("What the model learned")
        k = len(model["cats"])
        c = model["w"][1:1 + k]
        eff = [(n, 10 ** (v - c.mean())) for n, v in zip(model["cats"], c)]
        eff += [("Title +1 SD", 10 ** model["w"][1 + k]),
                ("Description +1 SD", 10 ** model["w"][2 + k]),
                ("Has link", 10 ** model["w"][3 + k])]
        hbar(pd.DataFrame(eff, columns=["Factor", "Effect (×)"]), "Factor", "Effect (×)", CORAL)
        st.caption("Effect on subscribers: each category vs. the average category, "
                   "title and description per 1 standard deviation, and having a link.")
