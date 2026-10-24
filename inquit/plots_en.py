#!/usr/bin/env python3
"""
plots_en.py — all figures for inquit_gender2.py and speech_who.py
with English titles, axis labels and captions.

Each figure is saved as <name>.png with its caption printed beneath it
(set EMBED_CAPTIONS = False to disable) and appended to captions.md.
`basis` = "morph" (gender from verb ending) or "tei" (gender from @who).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import chi2_contingency

EMBED_CAPTIONS = True
COL = {"masc": "#3b6ea8", "femn": "#c0504d"}
SEX = {"male": "masc", "female": "femn"}
SEXCOL = {"masc": "#3b6ea8", "femn": "#c0504d", "other": "#999999"}
MODE_ORDER = ["реплика", "вслух себе", "про себя"]
MODE_EN = {"реплика": "dialogue line", "вслух себе": "aloud to self",
           "про себя": "inner speech"}
CATEGORY_EN = {"нейтральные": "neutral", "вопрос": "question",
               "ответ/реакция": "reply / reaction", "продолжение": "continuation",
               "громкие": "loud", "тихие": "quiet",
               "эмоц./сопровождение": "emotional / accompanying",
               "комментарий": "comment", "внутр. речь": "inner speech",
               "не речь": "not speech", "—": "unlabelled"}
LAYOUT_EN = {"«»": "«…» (mostly inner speech)", "dash": "dash (dialogue)",
             "реплика": "dialogue line", "внутр./себе": "inner speech / to self"}
GENDER_EN = {"masc": "male", "femn": "female", "себе": "self", "—": "no addressee",
             "группа": "group", "?": "unknown", "plur": "plural", "neut": "neuter",
             "unk": "unknown"}


# ------------------------------------------------------------------ helpers
def theme():
    sns.set_theme(style="whitegrid", font="DejaVu Sans")   # DejaVu has Cyrillic


def glab(g, basis):
    if basis == "tei":
        return {"masc": "male speakers", "femn": "female speakers"}.get(g, GENDER_EN.get(g, g))
    return {"masc": "masculine verb forms",
            "femn": "feminine verb forms"}.get(g, GENDER_EN.get(g, g))


def gshort(g, basis):
    if basis == "tei":
        return {"masc": "male", "femn": "female"}.get(g, GENDER_EN.get(g, g))
    return {"masc": "masc. form", "femn": "fem. form"}.get(g, GENDER_EN.get(g, g))


def cat_en(c):
    return CATEGORY_EN.get(c, c)


def annot_pct(df):
    return np.where(df.values >= 0.5, df.round().astype(int).astype(str).values, "")


def reset_captions(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "captions.md").write_text("# Figure captions\n\n", encoding="utf-8")


def save(fig, out, name, title, caption):
    out = Path(out)
    cap = " ".join(caption.split())
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if EMBED_CAPTIONS:
        width = int(fig.get_size_inches()[0] * 14)
        fig.text(0.01, -0.01, textwrap.fill(cap, width), ha="left", va="top",
                 fontsize=9, style="italic")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(out / "captions.md", "a", encoding="utf-8") as fh:
        fh.write(f"**{name}. {title}.** {cap}\n\n")


def _part_key(p):
    try:
        return (0, int(p))
    except (TypeError, ValueError):
        return (1, str(p))


# ================================================================== gender plots (01–07)
def plot_top_verbs(t, out, top=20, basis="morph"):
    N = {g: int(t[g].sum()) for g in ("masc", "femn")}
    cover = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, .35 * top + 2.5), sharex=True)
    for ax, g in zip(axes, ("masc", "femn")):
        s = t.loc[t[g] > 0, f"pm_{g}"].nlargest(top)
        cover[g] = s.sum() / 10
        sns.barplot(x=s.values, y=s.index, ax=ax, color=COL[g])
        ax.set(title=f"{glab(g, basis).capitalize()} (n = {N[g]}; top {top} = "
                     f"{cover[g]:.0f}% of inquits)",
               xlabel="rate per 1,000 inquits", ylabel="")
    save(fig, out, "01_top_verbs", f"Top {top} inquit verbs",
         f"The most frequent verb lemmas in authorial inquits for {glab('masc', basis)} "
         f"(n = {N['masc']}) and {glab('femn', basis)} (n = {N['femn']}). Bars show rates "
         f"per 1,000 inquits of each group on a shared x-axis, so bar lengths are comparable "
         f"despite unequal sample sizes. The top {top} verbs cover {cover['masc']:.0f}% and "
         f"{cover['femn']:.0f}% of inquits respectively.")


def plot_scatter(t, out, min_freq=5, basis="morph", seed=0):
    rng = np.random.default_rng(seed)
    sh = t[t.status == "shared"]
    om_all, of_all = t[t.status == "only_masc"], t[t.status == "only_femn"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 8.5),
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    if len(sh):
        jit = lambda v: v * np.exp(rng.normal(0, .04, len(v)))
        x, y = jit(sh.pm_masc.to_numpy()), jit(sh.pm_femn.to_numpy())
        ax.scatter(x, y, s=8 * np.sqrt(sh.total), alpha=.6, color="#555")
        lo, hi = min(x.min(), y.min()) * .8, max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8, label="equal rate")
        ax.plot([lo, hi], [2 * lo, 2 * hi], ":", color=COL["femn"], lw=1,
                label=f"2× in {gshort('femn', basis)}")
        ax.plot([lo, hi], [lo / 2, hi / 2], ":", color=COL["masc"], lw=1,
                label=f"2× in {gshort('masc', basis)}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        texts = [ax.text(xi, yi, lem, fontsize=8)
                 for xi, yi, lem, n in zip(x, y, sh.index, sh.total) if n >= min_freq]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=.4, color="grey"))
        except ImportError:
            pass
        ax.legend(loc="upper left")
    ax.set(xscale="log", yscale="log",
           xlabel=f"{glab('masc', basis)}: rate per 1,000 inquits",
           ylabel=f"{glab('femn', basis)}: rate per 1,000 inquits",
           title=f"Shared verbs ({len(sh)} lemmas); point size ∝ √total frequency")

    om = om_all.masc.nlargest(12)
    of = of_all.femn.nlargest(12)
    ex = pd.concat([-om.sort_values(ascending=False), of.sort_values()])
    ax2.barh(ex.index, ex.values,
             color=[COL["masc"] if v < 0 else COL["femn"] for v in ex])
    ax2.axvline(0, color="k", lw=.6)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax2.set(xlabel=f"raw count   ← only {gshort('masc', basis)} | "
                   f"only {gshort('femn', basis)} →",
            title="Most frequent exclusive verbs")
    p_m = t.masc.sum() / (t.masc.sum() + t.femn.sum())
    n_m = int(np.ceil(np.log(.05) / np.log(p_m))) if 0 < p_m < 1 else 0
    n_f = int(np.ceil(np.log(.05) / np.log(1 - p_m))) if 0 < p_m < 1 else 0
    save(fig, out, "02_scatter_exclusive", "Shared and exclusive inquit verbs",
         f"Left: verbs attested for both groups, rates on log scales; the dashed line marks "
         f"equal rates, dotted lines a twofold difference. A small multiplicative jitter "
         f"separates identical points; only lemmas with ≥ {min_freq} occurrences are labelled. "
         f"Right: verbs found in one group only, raw counts. {len(om_all)} lemmas occur only "
         f"in {glab('masc', basis)} ({int((om_all.masc == 1).sum())} of them once) and "
         f"{len(of_all)} only in {glab('femn', basis)} ({int((of_all.femn == 1).sum())} once). "
         f"Given the {p_m:.0%} / {1 - p_m:.0%} split, exclusivity is unlikely by chance "
         f"(p < .05) only from ≈{n_m} occurrences for a masculine-only and ≈{n_f} for a "
         f"feminine-only verb.")


def plot_diff(t, out, min_freq=5, basis="morph"):
    d = t[t.total >= min_freq].sort_values("diff_pm")
    if d.empty:
        return
    sig = d.q_bh.lt(.05).to_numpy() if "q_bh" in d else np.zeros(len(d), bool)
    excl = ((d.diff_lo > 0) | (d.diff_hi < 0)).to_numpy()
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, .28 * len(d) + 2.5))
    ax.errorbar(d.diff_pm, y, fmt="none", ecolor="grey", lw=1,
                xerr=[np.clip(d.diff_pm - d.diff_lo, 0, None),
                      np.clip(d.diff_hi - d.diff_pm, 0, None)])
    ax.scatter(d.diff_pm, y, zorder=3, s=np.where(sig, 60, 20),
               c=[COL["femn"] if v > 0 else COL["masc"] for v in d.diff_pm],
               edgecolor="k", linewidths=np.where(sig, .8, 0))
    ax.set_yticks(y, d.index)
    ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel=f"difference in rate per 1,000 inquits "
                  f"({gshort('femn', basis)} − {gshort('masc', basis)})", ylabel="",
           title=f"Verbs with ≥ {min_freq} occurrences")
    save(fig, out, "03_difference", "Per-verb difference between the groups",
         f"Dots: difference in rate per 1,000 inquits; positive values = more typical of "
         f"{glab('femn', basis)}. Bars: 95% bootstrap intervals obtained by resampling whole "
         f"chapters, which respects clustering of inquits by scene. Large outlined dots: "
         f"q < .05 (Fisher's exact test, Benjamini–Hochberg correction). {int(sig.sum())} verbs "
         f"are significant, {int(excl.sum())} have intervals excluding zero, "
         f"{int((sig & excl).sum())} meet both criteria. The scale is absolute and favours "
         f"frequent verbs; see log2_ratio in verb_table.csv for relative differences.")


def plot_categories(mf, out, basis="morph"):
    lab = mf[~mf.category.isin(["—", "не речь"])]
    if lab.category.nunique() < 2:
        return
    tab = pd.crosstab(lab.category, lab.gender).reindex(columns=["masc", "femn"],
                                                         fill_value=0)
    tab = tab[tab.sum(1) > 0]
    chi2, p, dof, exp = chi2_contingency(tab)
    resid = (tab - exp) / np.sqrt(exp)
    share = tab / tab.sum() * 100
    ren = dict(index=cat_en, columns=lambda g: gshort(g, basis))
    fig, axes = plt.subplots(1, 2, figsize=(13, .45 * len(tab) + 3))
    sns.heatmap(share.rename(**ren), annot=True, fmt=".1f", cmap="Blues", ax=axes[0],
                cbar_kws={"label": "% of the group's inquits"})
    sns.heatmap(resid.rename(**ren), annot=True, fmt=".1f", cmap="coolwarm", center=0,
                ax=axes[1], cbar_kws={"label": "standardised residual"})
    axes[0].set(title="Share of each category, %", xlabel="", ylabel="")
    axes[1].set(title=f"Standardised residuals (χ² = {chi2:.1f}, df = {dof}, p = {p:.3g})",
                xlabel="", ylabel="")
    save(fig, out, "04_categories", "Semantic categories of inquit verbs",
         f"Left: share of each manually assigned verb category within the inquits of "
         f"{glab('masc', basis)} and {glab('femn', basis)} (unlabelled and 'not speech' "
         f"lemmas excluded; n = {int(tab.values.sum())}). Right: standardised residuals; "
         f"|r| > 2 marks a notable departure from independence. With two columns the residuals "
         f"in a row have opposite signs and the smaller group's residual is larger in absolute "
         f"value. {int((exp < 5).sum())} cell(s) have expected counts below 5.")


def plot_rarefaction(mf, out, basis="morph", reps=100):
    rng = np.random.default_rng(42)
    sizes_g = mf.gender.value_counts().reindex(["masc", "femn"]).fillna(0).astype(int)
    n_common = int(sizes_g.min())
    if n_common < 10:
        return
    at = {}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g in ("masc", "femn"):
        pool = mf.loc[mf.gender == g, "lemma"].to_numpy()
        sizes = np.unique(np.linspace(10, len(pool), 30).astype(int))
        sims = np.array([[len(set(rng.choice(pool, s, replace=False))) for _ in range(reps)]
                         for s in sizes])
        ax.plot(sizes, sims.mean(1), color=COL[g], label=f"{glab(g, basis)} (n = {len(pool)})")
        ax.fill_between(sizes, *np.percentile(sims, [2.5, 97.5], axis=1),
                        color=COL[g], alpha=.2)
        c = [len(set(rng.choice(pool, n_common, replace=False))) for _ in range(reps)]
        at[g] = (np.mean(c), *np.percentile(c, [2.5, 97.5]))
    ax.axvline(n_common, color="grey", ls=":")
    ax.set(xlabel="number of inquits sampled", ylabel="distinct verb lemmas",
           title=f"Comparison point: n = {n_common}")
    ax.legend()
    save(fig, out, "05_rarefaction", "Rarefaction curves of inquit verbs",
         f"Expected number of distinct verb lemmas in random subsamples of increasing size "
         f"(mean of {reps} draws; shaded 95% band of subsampling variation, which narrows to "
         f"zero at the full sample). At the common size n = {n_common} (dotted line): "
         f"{glab('masc', basis)} {at['masc'][0]:.1f} [{at['masc'][1]:.0f}–{at['masc'][2]:.0f}], "
         f"{glab('femn', basis)} {at['femn'][0]:.1f} [{at['femn'][1]:.0f}–{at['femn'][2]:.0f}]. "
         f"Raw type counts are not comparable because they grow with sample size.")


def plot_confounds(df, mf, out, basis="morph"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    order = [g for g in ["masc", "femn", "neut", "plur", "unk"] if g in set(df.gender)]
    ct = pd.crosstab(df.gender, df.layout).reindex(order).fillna(0)
    ct.rename(index=lambda g: gshort(g, basis), columns=lambda c: LAYOUT_EN.get(c, c)) \
      .plot(kind="bar", stacked=True, ax=axes[0], color=["#8fb3d9", "#e3a857", "#999"][:ct.shape[1]])
    axes[0].set(title="Inquits by group and type of utterance", xlabel="", ylabel="inquits")
    axes[0].tick_params(axis="x", rotation=0)

    by = pd.crosstab(mf.part, mf.gender).reindex(columns=["masc", "femn"], fill_value=0)
    by = by.loc[sorted(by.index, key=_part_key)]
    n = by.sum(1); pf = by.femn / n
    se = np.sqrt(pf * (1 - pf) / n)
    x = np.arange(len(by))
    axes[1].errorbar(x, pf, yerr=1.96 * se, fmt="o-", color=COL["femn"], capsize=3)
    overall = by.femn.sum() / n.sum()
    axes[1].axhline(overall, color="grey", ls="--", lw=.8)
    for xi, yi, ni in zip(x, pf, n):
        axes[1].annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(5, 6),
                         fontsize=8)
    axes[1].set_xticks(x, by.index)
    axes[1].set(title=f"Share of {glab('femn', basis)} by part of the novel",
                xlabel="part", ylabel="share")
    save(fig, out, "06_confounds", "Possible confounds",
         f"Left: counts of inquits by group and utterance type (in run A «…»-framed vs "
         f"dash-introduced utterances; with @who, dialogue lines vs inner speech / speech to "
         f"oneself). Right: share of {glab('femn', basis)} among masculine+feminine inquits in "
         f"each part (±1.96 SE, an optimistic interval since inquits cluster by scene); dashed "
         f"line = overall share ({overall:.0%}). Parts far above the line dominate the "
         f"feminine profile.")


# def plot_manner(df, out, stop=frozenset(), basis="morph", top=15):
#     mf = df[df.gender.isin(["masc", "femn"])]
#     N = mf.gender.value_counts().reindex(["masc", "femn"]).fillna(0)
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     vd, n_types = None, {}
#     for ax, col, name in zip(axes, ("adverbs", "gerunds"),
#                              ("Manner adverbs", "Gerunds (деепричастия)")):
#         if col not in mf:
#             ax.axis("off"); continue
#         s = (mf[["gender"]].assign(w=mf[col].fillna("").astype(str).str.split())
#              .explode("w").dropna())
#         s = s[s.w != ""]
#         if col == "adverbs":
#             vd = (s.w == "вдруг").groupby(s.gender).sum().reindex(N.index, fill_value=0) / N * 1000
#             s = s[~s.w.isin(stop)]
#         tab = pd.crosstab(s.w, s.gender).reindex(columns=N.index, fill_value=0)
#         n_types[col] = len(tab)
#         if tab.empty:
#             ax.axis("off"); continue
#         pm = tab.div(N, axis=1) * 1000
#         sel = tab.sum(1).nlargest(top).index
#         long = (pm.loc[sel].rename(columns=lambda g: gshort(g, basis)).rename_axis("w")
#                 .reset_index().melt(id_vars="w", var_name="group", value_name="pm"))
#         sns.barplot(data=long, y="w", x="pm", hue="group", ax=ax,
#                     hue_order=[gshort("masc", basis), gshort("femn", basis)],
#                     palette=[COL["masc"], COL["femn"]])
#         ax.set(xlabel="rate per 1,000 inquits", ylabel="", title=f"{name} (top {top})")
#         ax.legend(title="")
#     vtxt = (f"The discourse marker вдруг, excluded from the left panel, occurs at "
#             f"{vd['masc']:.1f}‰ ({gshort('masc', basis)}) vs {vd['femn']:.1f}‰ "
#             f"({gshort('femn', basis)}).") if vd is not None else ""
#     save(fig, out, "07_manner", "Manner modifiers of the speech verb",
#          f"Adverbs (advmod) and gerunds syntactically dependent on the inquit verb according "
#          f"to the spaCy parse, as rates per 1,000 inquits. Temporal and discourse adverbs "
#          f"(опять, тут, еще …) are excluded from the adverb panel. {vtxt} Most items are rare; "
#          f"compare semantic groups (volume, attitude, gesture) rather than single words.")


def plot_rest(df, t, mf, out, top=20, basis="morph"):
    plot_top_verbs(t, out, top, basis)
    plot_categories(mf, out, basis)
    plot_rarefaction(mf, out, basis)
    plot_confounds(df, mf, out, basis)