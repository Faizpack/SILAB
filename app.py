"""YouTube Dashboard + statistical analysis (Streamlit).

Pages (left sidebar): Dashboard, Chi-square, Two-way ANOVA, CLT,
Covariance & correlation, Regression.

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
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

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


# ------------------------------------------------------- derived columns
def prep(df):
    d = df.copy()
    d["logsubs"] = np.log10(d["subs"].where(d["subs"] > 0))
    d["lw"] = np.log10(1 + d["dw"])
    d["linkn"] = d["link"].astype(int)
    d["linklab"] = np.where(d["link"], "Has link", "No link")
    d["size"] = pd.cut(d["subs"], [0, 1e5, 1e6, 1e7, np.inf],
                       labels=["<100K", "100K–1M", "1M–10M", "10M+"], right=False)
    for src, dst in (("tl", "tgroup"), ("dw", "dgroup")):
        d[dst] = pd.qcut(d[src].rank(method="first"), 3, labels=["Short", "Medium", "Long"])
    return d


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def peq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"


def verdict(p, alpha, what):
    if p < alpha:
        return f"✅ {peq(p)} (below α = {alpha}): **{what}** (statistically significant)."
    return f"➖ {peq(p)} (not below α = {alpha}): **no significant evidence** that {what}."


def heatmap(m, fmt=".2f"):
    long = pd.DataFrame([(r, c, float(m.loc[r, c])) for r in m.index for c in m.columns],
                        columns=["row", "col", "v"])
    lim = max(float(long["v"].abs().max()), 1e-9)
    base = alt.Chart(long).encode(
        x=alt.X("col:N", sort=list(m.columns), title=None),
        y=alt.Y("row:N", sort=list(m.index), title=None))
    rect = base.mark_rect().encode(
        color=alt.Color("v:Q", scale=alt.Scale(scheme="redblue", domain=[-lim, lim]), legend=None),
        tooltip=["row", "col", alt.Tooltip("v:Q", format=fmt)])
    text = base.mark_text().encode(
        text=alt.Text("v:Q", format=fmt),
        color=alt.condition(f"abs(datum.v) > {0.6 * lim}", alt.value("white"), alt.value("black")))
    show((rect + text).properties(height=max(150, 46 * len(m.index))))


RESP = {"log10 subscribers": "logsubs", "Title length (characters)": "tl", "Description words": "dw"}


# ------------------------------------------------------------------ pages
def page_dashboard(data, current):
    cats = sorted(data["cat"].unique())
    sel = data if current == "All categories" else data[data["cat"] == current]
    with_s = sel.dropna(subset=["subs"])
    st.caption(f"Loaded {len(data):,} unique videos.")
    tab1, tab2, tab3 = st.tabs(["Overview", "Categories", "Prediction"])

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
                columns=["Size", "Videos"])
            hbar(hist, "Size", "Videos", GOLD, order=[e[0] for e in edges])
        with right:
            st.subheader("Top channels")
            top = (sel.dropna(subset=["subs"]).groupby("channel", as_index=False)
                   .agg(Category=("cat", "first"), Subscribers=("subs", "first"), Videos=("channel", "size"))
                   .sort_values("Subscribers", ascending=False).head(10)
                   .rename(columns={"channel": "Channel"}))
            top["Subscribers"] = top["Subscribers"].map(fmt)
            st.dataframe(top, hide_index=True)

    with tab2:
        g = data.groupby("cat")
        sub = data.dropna(subset=["subs"]).groupby("cat")["subs"]
        per = pd.DataFrame({
            "Videos": g.size(),
            "Median subscribers": sub.median(),
            "Share from 1M+ (%)": sub.apply(lambda s: (s > 1e6).mean() * 100),
            "Avg title length": g["tl"].mean(),
            "With a link (%)": g["link"].mean() * 100,
        }).reset_index().rename(columns={"cat": "Category"})
        plots = [("Videos per category", "Videos", TEAL),
                 ("Median subscribers", "Median subscribers", GOLD),
                 ("Videos from 1M+ channels (%)", "Share from 1M+ (%)", CORAL),
                 ("Average title length", "Avg title length", TEAL),
                 ("Videos with a link in description (%)", "With a link (%)", CORAL)]
        for (title, colname, color), col in zip(plots, [c for _ in range(3) for c in st.columns(2)]):
            with col:
                st.subheader(title)
                hbar(per[["Category", colname]].fillna(0), "Category", colname, color)

    with tab3:
        st.subheader("Prediction model: how big is the channel?")
        model = train_model(data)
        if model is None:
            st.warning("Not enough rows with subscriber counts to train a model.")
            return
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
        st.caption(f"Ridge regression on log10(subscribers). Trained on {model['n_train']:,} videos, "
                   f"tested on {model['n_test']:,} unseen ones. Test R² = {model['r2']:.2f}; "
                   f"a typical miss is about {model['fold']:.1f}× in subscribers. "
                   "Treat it as a rough guide, not a forecast.")
        st.subheader("What the model learned")
        k = len(model["cats"])
        c = model["w"][1:1 + k]
        eff = [(n, 10 ** (v - c.mean())) for n, v in zip(model["cats"], c)]
        eff += [("Title +1 SD", 10 ** model["w"][1 + k]),
                ("Description +1 SD", 10 ** model["w"][2 + k]),
                ("Has link", 10 ** model["w"][3 + k])]
        hbar(pd.DataFrame(eff, columns=["Factor", "Effect (×)"]), "Factor", "Effect (×)", CORAL)


def page_chi(df):
    st.header("Chi-square test of independence")
    st.write("Tests whether two categorical variables are related. "
             "**H₀:** the variables are independent. **H₁:** they are associated.")
    V = {"Category": "cat", "Channel size": "size", "Title length group": "tgroup",
         "Description length group": "dgroup", "Link in description": "linklab"}
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Variable 1 (rows)", list(V), 0)
    b = c2.selectbox("Variable 2 (columns)", list(V), 1)
    alpha = c3.select_slider("Significance level α", [0.01, 0.05, 0.10], 0.05)
    if a == b:
        st.warning("Pick two different variables.")
        return
    sub = df[[V[a], V[b]]].dropna()
    ct = pd.crosstab(sub[V[a]], sub[V[b]])
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        st.warning("Need at least 2 levels in each variable.")
        return
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cv = sqrt(chi2 / (n * (min(ct.shape) - 1)))
    strength = "negligible" if cv < .1 else "small" if cv < .3 else "moderate" if cv < .5 else "strong"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("χ² statistic", f"{chi2:,.2f}")
    m2.metric("Degrees of freedom", dof)
    m3.metric("p-value", fp(p))
    m4.metric("Cramér's V", f"{cv:.3f}", strength, delta_color="off")
    st.markdown(verdict(p, alpha, f"{a} and {b} are associated"))
    low = (exp < 5).mean()
    if low > 0.2:
        st.warning(f"{low:.0%} of cells have expected counts below 5, so the chi-square approximation may be unreliable.")
    t1, t2, t3 = st.tabs(["Observed", "Expected", "Standardised residuals"])
    t1.dataframe(ct)
    t2.dataframe(pd.DataFrame(exp, index=ct.index, columns=ct.columns).round(1))
    with t3:
        resid = pd.DataFrame((ct.values - exp) / np.sqrt(exp), index=ct.index, columns=ct.columns)
        heatmap(resid)
        st.caption("Values beyond ±2 show cells with notably more (+) or fewer (−) videos than independence predicts.")


def page_anova(df):
    st.header("Two-way ANOVA")
    st.write("Compares group means for two factors at once and tests for an **interaction** "
             "(does the effect of one factor depend on the other?). Factor A is always **Category**.")
    FB = {"Link in description": "linklab", "Title length group": "tgroup", "Description length group": "dgroup"}
    c1, c2, c3 = st.columns(3)
    fb = c2.selectbox("Factor B", list(FB))
    blocked = {"tgroup": "tl", "dgroup": "dw"}.get(FB[fb])
    ropts = {k: v for k, v in RESP.items() if v != blocked}
    rn = c1.selectbox("Response (compared across groups)", list(ropts))
    alpha = c3.select_slider("Significance level α", [0.01, 0.05, 0.10], 0.05)
    d = df[[ropts[rn], "cat", FB[fb]]].dropna()
    d.columns = ["Y", "A", "B"]
    d["A"], d["B"] = d["A"].astype(str), d["B"].astype(str)
    cnt = pd.crosstab(d["A"], d["B"])
    keep = cnt.index[(cnt >= 2).all(axis=1)]
    dropped = sorted(set(cnt.index) - set(keep))
    d = d[d["A"].isin(keep)]
    if d["A"].nunique() < 2 or d["B"].nunique() < 2:
        st.warning("Not enough groups with data for a two-way ANOVA.")
        return
    if dropped:
        st.caption("Categories left out (fewer than 2 videos in some cell): " + ", ".join(dropped))
    model = smf.ols("Y ~ C(A) * C(B)", data=d).fit()
    tbl = anova_lm(model, typ=2)
    ss_res = tbl.loc["Residual", "sum_sq"]
    names = {"C(A)": "Category", "C(B)": fb, "C(A):C(B)": f"Category × {fb}", "Residual": "Residual"}
    out = pd.DataFrame({
        "Source": [names[i] for i in tbl.index],
        "SS": tbl["sum_sq"].round(3).to_numpy(),
        "df": tbl["df"].astype(int).to_numpy(),
        "MS": (tbl["sum_sq"] / tbl["df"]).round(3).to_numpy(),
        "F": tbl["F"].round(3).to_numpy(),
        "p-value": [fp(v) if pd.notna(v) else "" for v in tbl["PR(>F)"]],
        "Partial η²": [round(s / (s + ss_res), 3) if i != "Residual" else np.nan
                       for i, s in zip(tbl.index, tbl["sum_sq"])],
    })
    st.dataframe(out, hide_index=True)
    for idx in ["C(A)", "C(B)", "C(A):C(B)"]:
        p = tbl.loc[idx, "PR(>F)"]
        what = {"C(A)": f"{rn} differs across categories",
                "C(B)": f"{rn} differs by {fb.lower()}",
                "C(A):C(B)": f"the effect of {fb.lower()} depends on category (interaction)"}[idx]
        st.markdown(verdict(p, alpha, what))
    st.subheader("Interaction plot (group means)")
    gm = d.groupby(["A", "B"])["Y"].agg(["mean", "sem"]).reset_index()
    gm["lo"], gm["hi"] = gm["mean"] - gm["sem"].fillna(0), gm["mean"] + gm["sem"].fillna(0)
    base = alt.Chart(gm).encode(x=alt.X("A:N", title="Category"), color=alt.Color("B:N", title=fb))
    show((base.mark_line().encode(y=alt.Y("mean:Q", title=f"Mean {rn} (± SE)", scale=alt.Scale(zero=False)))
          + base.mark_point(filled=True, size=60).encode(y="mean:Q", tooltip=["A", "B", alt.Tooltip("mean:Q", format=".2f")])
          + base.mark_rule().encode(y="lo:Q", y2="hi:Q")).properties(height=320))
    st.caption("Roughly parallel lines suggest no interaction; crossing or diverging lines suggest one.")
    groups = [g["Y"].to_numpy() for _, g in d.groupby(["A", "B"]) if len(g) > 1]
    lev = stats.levene(*groups)
    res = model.resid.sample(min(5000, len(d)), random_state=1)
    sh = stats.shapiro(res)
    st.caption(f"Assumption checks: Levene (equal variances) p = {fp(lev.pvalue)}; Shapiro–Wilk on residuals "
               f"(normality) p = {fp(sh.pvalue)}. With many videos ANOVA is fairly robust, but small p-values "
               "here mean the F-tests should be read with some caution.")


def page_clt(df):
    st.header("Central Limit Theorem")
    st.write("Even when the original data are skewed, the **means of repeated random samples** "
             "become approximately normal as the sample size grows, with SD ≈ σ/√n.")
    V = {"Subscribers (raw)": "subs", "Title length": "tl", "Description words": "dw"}
    c1, c2, c3, c4 = st.columns(4)
    name = c1.selectbox("Variable", list(V))
    n = c2.slider("Sample size n", 1, 200, 30)
    k = c3.slider("Number of samples", 100, 5000, 1000, 100)
    seed = c4.number_input("Random seed", 0, 9999, 42)
    x = df[V[name]].dropna().to_numpy(float)
    if len(x) < 10:
        st.warning("Not enough data.")
        return
    rng = np.random.default_rng(int(seed))
    means = rng.choice(x, size=(k, n), replace=True).mean(axis=1)
    mu, sigma = x.mean(), x.std(ddof=0)
    se = sigma / np.sqrt(n)

    def hist(vals, lo, hi):
        cnt, ed = np.histogram(np.clip(vals, lo, hi), bins=40, range=(lo, hi), density=True)
        return pd.DataFrame({"x0": ed[:-1], "x1": ed[1:], "density": cnt})

    def hchart(h, color):
        return alt.Chart(h).mark_bar(color=color).encode(
            x=alt.X("x0:Q", title=name), x2="x1:Q", y=alt.Y("density:Q", title="Density"))

    left, right = st.columns(2)
    with left:
        st.subheader("Population (all videos)")
        hi = float(np.percentile(x, 99))
        show(hchart(hist(x, float(x.min()), hi), GOLD).properties(height=300))
        st.caption(f"Skewness = {stats.skew(x):.2f}. Top 1% of values are not shown (axis cut at {fmt(hi)}).")
    with right:
        st.subheader(f"Distribution of {k:,} sample means (n = {n})")
        lo2, hi2 = float(means.min()), float(means.max())
        xs = np.linspace(lo2, hi2, 200)
        curve = pd.DataFrame({"x": xs, "pdf": stats.norm.pdf(xs, mu, se)})
        line = alt.Chart(curve).mark_line(color=CORAL, strokeWidth=2).encode(x="x:Q", y="pdf:Q")
        show((hchart(hist(means, lo2, hi2), TEAL) + line).properties(height=300))
        st.caption("Red curve = normal distribution predicted by the CLT.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population mean μ", fmt(mu) if V[name] == "subs" else f"{mu:.2f}")
    m2.metric("Mean of sample means", fmt(means.mean()) if V[name] == "subs" else f"{means.mean():.2f}")
    m3.metric("Theoretical SE = σ/√n", fmt(se) if V[name] == "subs" else f"{se:.2f}")
    m4.metric("Observed SD of means", fmt(means.std(ddof=1)) if V[name] == "subs" else f"{means.std(ddof=1):.2f}")
    st.subheader("How the shape changes with sample size")
    rows = []
    for nn in (1, 2, 5, 10, 30, 100):
        mm = rng.choice(x, size=(1000, nn), replace=True).mean(axis=1)
        rows.append({"n": nn, "Skewness of means": round(float(stats.skew(mm)), 3),
                     "Excess kurtosis": round(float(stats.kurtosis(mm)), 3),
                     "Shapiro–Wilk p": fp(stats.shapiro(mm).pvalue)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Skewness and kurtosis moving toward 0 mean the sample means are looking more normal.")


def page_corr(df):
    st.header("Covariance & correlation")
    V = {"log10 subscribers": "logsubs", "Subscribers (raw)": "subs", "Title length": "tl",
         "Description words": "dw", "Has link (0/1)": "linkn"}
    chosen = st.multiselect("Variables", list(V), ["log10 subscribers", "Title length", "Description words", "Has link (0/1)"])
    d = df[[V[c] for c in chosen]].dropna() if chosen else pd.DataFrame()
    d = d.rename(columns={v: k for k, v in V.items()})
    d = d.loc[:, d.std() > 0] if len(d) else d
    if d.shape[1] < 2:
        st.warning("Pick at least two variables that vary.")
        return
    t1, t2, t3 = st.tabs(["Pearson correlation", "Spearman correlation", "Covariance"])
    with t1:
        heatmap(d.corr())
        st.caption("Pearson r measures the strength of a straight-line relationship (−1 to +1).")
    with t2:
        heatmap(d.corr(method="spearman"))
        st.caption("Spearman ρ uses ranks, so it is robust to outliers and skew (useful for raw subscribers).")
    with t3:
        st.dataframe(d.cov().style.format("{:,.3f}"))
        st.caption("Covariance shows the direction of the relationship, but its size depends on the units, "
                   "so compare variables with correlation instead.")
    st.subheader("Pair analysis")
    c1, c2 = st.columns(2)
    xv = c1.selectbox("X variable", list(d.columns), 0)
    yv = c2.selectbox("Y variable", list(d.columns), min(1, len(d.columns) - 1))
    if xv == yv:
        st.info("Choose two different variables.")
        return
    pr, pp = stats.pearsonr(d[xv], d[yv])
    sr, sp = stats.spearmanr(d[xv], d[yv])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Covariance", f"{np.cov(d[xv], d[yv])[0, 1]:,.3f}")
    m2.metric("Pearson r", f"{pr:.3f}", f"p {fp(pp)}", delta_color="off")
    m3.metric("Spearman ρ", f"{sr:.3f}", f"p {fp(sp)}", delta_color="off")
    m4.metric("R² (r squared)", f"{pr ** 2:.3f}")
    pts = d[[xv, yv]].sample(min(3000, len(d)), random_state=1)
    sc = alt.Chart(pts).mark_circle(opacity=.35, color=TEAL).encode(
        x=alt.X(xv, scale=alt.Scale(zero=False)), y=alt.Y(yv, scale=alt.Scale(zero=False)))
    show((sc + sc.transform_regression(xv, yv).mark_line(color=CORAL, strokeWidth=3)).properties(height=360))
    st.caption("Correlation does not prove causation. Plot shows up to 3,000 random videos.")


def page_reg(df):
    st.header("Regression analysis (OLS)")
    st.write("Multiple linear regression: how much each factor changes the response while holding the others fixed.")
    c1, c2 = st.columns([1, 2])
    rn = c1.selectbox("Response (Y)", list(RESP))
    resp = RESP[rn]
    P = {"Title length": "tl", "Description words (log10)": "lw", "Has link": "linkn", "Category": "C(cat)"}
    if resp == "tl":
        P.pop("Title length")
    if resp == "dw":
        P.pop("Description words (log10)")
    chosen = c2.multiselect("Predictors (X)", list(P), list(P))
    if not chosen:
        st.warning("Choose at least one predictor.")
        return
    d = df.dropna(subset=[resp]).copy()
    d["Y"] = d[resp]
    model = smf.ols("Y ~ " + " + ".join(P[c] for c in chosen), data=d).fit()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R²", f"{model.rsquared:.3f}")
    m2.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
    m3.metric("F-test p-value", fp(model.f_pvalue))
    m4.metric("RMSE", f"{np.sqrt(model.mse_resid):.3f}")
    m5.metric("Observations", f"{int(model.nobs):,}")
    ci = model.conf_int()
    coef = pd.DataFrame({
        "Term": [t.replace("C(cat)[T.", "Category: ").rstrip("]") for t in model.params.index],
        "Coefficient": model.params.round(4).to_numpy(),
        "Std. error": model.bse.round(4).to_numpy(),
        "t": model.tvalues.round(2).to_numpy(),
        "p-value": [fp(v) for v in model.pvalues],
        "95% CI": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(ci[0], ci[1])],
        "Significant": ["Yes" if v < 0.05 else "No" for v in model.pvalues],
    })
    if resp == "logsubs":
        coef["Effect on subs (×)"] = (10 ** model.params).round(3).to_numpy()
    st.subheader("Coefficients")
    st.dataframe(coef, hide_index=True)
    st.caption("Each coefficient is the expected change in Y for a 1-unit rise in that predictor "
               "(or versus the reference category). For log10 subscribers, 10^coefficient is the multiplier on subscribers.")
    fit = pd.DataFrame({"Fitted": model.fittedvalues, "Actual": d["Y"], "Residual": model.resid})
    fit = fit.sample(min(3000, len(fit)), random_state=1)
    l, r = st.columns(2)
    with l:
        st.subheader("Actual vs fitted")
        sc = alt.Chart(fit).mark_circle(opacity=.35, color=TEAL).encode(
            x=alt.X("Fitted", scale=alt.Scale(zero=False)), y=alt.Y("Actual", scale=alt.Scale(zero=False)))
        show((sc + sc.transform_regression("Fitted", "Actual").mark_line(color=CORAL)).properties(height=320))
    with r:
        st.subheader("Residuals vs fitted")
        sc = alt.Chart(fit).mark_circle(opacity=.35, color=GOLD).encode(
            x=alt.X("Fitted", scale=alt.Scale(zero=False)), y="Residual")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CORAL).encode(y="y:Q")
        show((sc + zero).properties(height=320))
    st.caption("A good fit shows residuals scattered evenly around 0 with no curve or funnel shape.")


# -------------------------------------------------------------------- UI
st.title("📊 YouTube Dashboard")
PAGES = ["Dashboard", "Chi-square test", "Two-way ANOVA", "Central Limit Theorem",
         "Covariance & correlation", "Regression analysis"]

with st.sidebar:
    page = st.radio("Analysis", PAGES)
    st.divider()
    st.header("Data")
    up = st.file_uploader("Choose youtube.csv", type="csv")
    st.caption("The file is processed locally by this app.")

raw = up.getvalue() if up else (DEFAULT_CSV.read_bytes() if DEFAULT_CSV.exists() else None)
if raw is None:
    st.info("Upload youtube.csv in the sidebar to start.")
    st.stop()
try:
    data = prep(load_data(raw))
except Exception as e:  # noqa: BLE001
    st.error(str(e) if isinstance(e, ValueError) else "Could not read this file.")
    st.stop()

current = "All categories"
if page in ("Dashboard", "Central Limit Theorem", "Covariance & correlation"):
    with st.sidebar:
        st.header("Filter")
        current = st.selectbox("Category", ["All categories", *sorted(data["cat"].unique())])
scoped = data if current == "All categories" else data[data["cat"] == current]

if page == "Dashboard":
    page_dashboard(data, current)
elif page == "Chi-square test":
    page_chi(data)
elif page == "Two-way ANOVA":
    page_anova(data)
elif page == "Central Limit Theorem":
    page_clt(scoped)
elif page == "Covariance & correlation":
    page_corr(scoped)
else:
    page_reg(data)
