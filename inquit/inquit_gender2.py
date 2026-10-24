#!/usr/bin/env python3
"""
inquit_gender.py — speech verbs in the author's words (inquits) of the
Digital Dostoevsky TEI markup of "Crime and Punishment", split by grammatical gender.

1) extracts inquits after/before <said>, lemmatises them, picks the speech verb,
   determines gender (past-tense morphology -> subject fallback);
2) sorts and compares the masculine and feminine lists;
3) plots the results.
"""
from __future__ import annotations

import argparse
import logging
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy3
import seaborn as sns
import spacy
from lxml import etree
from scipy.stats import chi2_contingency, fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

TEI = "{http://www.tei-c.org/ns/1.0}"
SAID, PERS, P = TEI + "said", TEI + "persName", TEI + "p"
morph = pymorphy3.MorphAnalyzer()

# ------------------------------------------------------------------ lexicon
SPEECH_LEXICON: dict[str, set[str]] = {
    "нейтральные": {"сказать", "говорить", "проговорить", "произнести", "произносить",
                    "промолвить", "вымолвить", "молвить", "выговорить"},
    "вопрос": {"спросить", "спрашивать", "переспросить", "осведомиться", "допрашивать"},
    "ответ/реакция": {"ответить", "отвечать", "возразить", "возражать", "подтвердить",
                      "согласиться", "отрезать", "огрызнуться", "отозваться", "откликнуться", "откликаться"},
    "продолжение": {"продолжать", "продолжить", "прибавить", "прибавлять", "добавить",
                    "подхватить", "перебить", "перебивать", "заключить", "докончить",
                    "повторить", "повторять"},
    "громкие": {"крикнуть", "кричать", "вскрикнуть", "закричать", "прокричать",
                "воскликнуть", "восклицать", "заорать", "орать", "завопить", "вопить",
                "взвизгнуть", "визжать", "рявкнуть", "гаркнуть"},
    "тихие": {"шептать", "прошептать", "шепнуть", "зашептать", "бормотать",
              "пробормотать", "забормотать", "пролепетать", "лепетать", "пробурчать",
              "прохрипеть"},
    "эмоц./сопровождение": {"простонать", "застонать", "всхлипнуть", "зарыдать",
                            "засмеяться", "рассмеяться", "усмехнуться", "улыбнуться",
                            "проворчать", "ворчать", "вздохнуть", "захохотать", "вспыхнуть"},
    "комментарий": {"заметить", "замечать", "пояснить", "объяснить", "объявить",
                    "сообщить", "уверять", "настаивать", "обратиться", "предложить",
                    "заявить", "признаться"},
    "внутр. речь": {"подумать", "думать", "соображать", "сообразить", "мелькнуть",
                    "решить", "подуматься", "мечтать"},
}

PHASE_VERBS = {"поспешить", "начать", "стать", "хотеть", "захотеть", "попытаться",
               "пытаться", "решиться", "успеть", "продолжать", "силиться"}

PRON_GENDER = {"он": "masc", "она": "femn", "оно": "neut", "они": "plur"}
UD2PM = {"Masc": "masc", "Fem": "femn", "Neut": "neut"}
NAME_GENDER_OVERRIDES = {  # names pymorphy3 may not tag as Name/Surn — extend as needed
    "Миколка": "masc", "Миколай": "masc", "Порох": "masc", "Кох": "masc",
    "Липпевехзель": "femn", "Ресслих": "femn", "Лавиза": "femn", "Амаль": "femn",
    "Амаль-Иван": "femn", "Настасья": "femn", "Пашенька": "femn",
}

# ------------------------------------------------------------------ extraction
POST_RE = re.compile(r"^\s*[,.!?…;]*\s*—\s*(.+)", re.S)
CUT_RE = re.compile(r"[.!?…:;]|\s—")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def text_of(el) -> str:
    return "".join(el.itertext())


def last_sentence(text: str) -> str:
    return SENT_SPLIT.split(text.strip())[-1]


def iter_paragraphs(root):
    if root.find(f".//{TEI}div1") is not None:            # old sample structure
        for div1 in root.iter(TEI + "div1"):
            for div3 in div1.iter(TEI + "div3"):
                for p in div3.iter(P):
                    yield div1.get("n"), div3.get("n"), p
        return
    for part in root.iter(TEI + "div"):                     # full TEI structure
        if part.get("type") != "part":
            continue
        for sec in part.iter(TEI + "div"):
            if sec.get("type") == "section":
                for p in sec.iter(P):
                    yield part.get("n"), sec.get("n"), p

def make_row(part, chap, position, inquit, names, said):
    inquit = re.sub(r"\s+", " ", inquit).strip()
    return dict(part=part, chapter=chap, position=position, inquit=inquit,
                names=[n for n in names if n in inquit],
                said=said.strip()[:120],
                # «…» = embedded/inner speech in this edition; dash = dialogue line
                layout="«»" if said.lstrip().startswith("«") else "dash")


def extract_candidates(root) -> list[dict]:
    rows, prev_p_text = [], ""
    for part, chap, p in iter_paragraphs(root):
        children = list(p)
        # pre-inquit in the previous paragraph: "... произнесла, пропуская гостя вперед:"
        if (children and children[0].tag == SAID
                and (p.text or "").strip(" —\u00a0\t") == ""
                and prev_p_text.rstrip().endswith(":")):
            rows.append(make_row(part, chap, "pre", last_sentence(prev_p_text), [],
                                 text_of(children[0])))

        buffer, buffer_names = p.text or "", []
        for i, el in enumerate(children):
            if el.tag != SAID:
                buffer += text_of(el) + (el.tail or "")
                if el.tag == PERS:
                    buffer_names.append(text_of(el))
                continue
            said = text_of(el)
            # pre-inquit inside paragraph: "... крикнул ему вдруг, проезжая: <said>"
            if buffer.rstrip().endswith(":"):
                rows.append(make_row(part, chap, "pre", last_sentence(buffer),
                                     buffer_names, said))
            # post-inquit: "</said> — бормотал он в смущении, — <said>"
            seg, names = el.tail or "", []
            for sib in children[i + 1:]:
                if sib.tag == SAID:
                    break
                seg += text_of(sib) + (sib.tail or "")
                if sib.tag == PERS:
                    names.append(text_of(sib))
            m = POST_RE.match(seg)
            if m:
                body = m.group(1).lstrip()
                if body and body[0].islower():          # uppercase = new narrative sentence
                    rows.append(make_row(part, chap, "post", CUT_RE.split(body, 1)[0],
                                         names, said))
            buffer, buffer_names = el.tail or "", []
        prev_p_text = text_of(p)
    return rows


# ------------------------------------------------------------------ morphology
def best_verb_parse(word: str, spacy_pos: str):
    """pymorphy3 parse that is a finite verb / infinitive, cross-checked with spaCy POS."""
    for k, p in enumerate(morph.parse(word)):
        if p.tag.POS in ("VERB", "INFN") and (spacy_pos in ("VERB", "AUX") or k == 0):
            return p
    return None

def _g(x):
    """pymorphy3 grammeme object -> plain str (or None)."""
    return None if x is None else str(x)

def morph_gender(p):
    if p.tag.POS == "VERB" and p.tag.tense == "past":
        return "plur" if p.tag.number == "plur" else _g(p.tag.gender)
    return None

@lru_cache(maxsize=None)
def name_gender(name: str):
    for tok in name.split():
        if tok in NAME_GENDER_OVERRIDES:
            return NAME_GENDER_OVERRIDES[tok]
        for p in morph.parse(tok):
            if p.tag.grammemes & {"Name", "Surn", "Patr"} and p.tag.gender:
                return _g(p.tag.gender)
    return None

def token_gender(t, names):
    if t.lower_ in PRON_GENDER:
        return PRON_GENDER[t.lower_]
    for n in names:
        if t.text in n.split():
            return name_gender(n)
    if t.pos_ in ("NOUN", "PROPN"):
        g = t.morph.get("Gender")
        if g:
            return UD2PM.get(g[0])
        for p in morph.parse(t.text):
            if p.tag.POS == "NOUN" and "nomn" in p.tag:
                return _g(p.tag.gender)
    return None


def subject_gender(doc, verb_tok, names):
    for ch in verb_tok.children:                       # 1) dependency: nsubj
        if ch.dep_.startswith("nsubj"):
            g = token_gender(ch, names)
            if g:
                return g
    for t in doc:                                      # 2) linear fallback
        if t.lower_ in PRON_GENDER or (t.pos_ in ("NOUN", "PROPN")
                                       and "Nom" in t.morph.get("Case")):
            g = token_gender(t, names)
            if g:
                return g
    return None

# ======================================================= 1. verb choice without a lexicon
NON_MANNER_ADV = {
    "вдруг", "опять", "снова", "вновь", "тут", "еще", "ещё", "уже", "вот", "потом",
    "теперь", "наконец", "тоже", "также", "так", "затем", "сейчас", "тотчас", "почти",
    "даже", "только", "здесь", "там", "сюда", "туда", "всегда", "никогда", "вовсе",
}
SEED_CATEGORIES = {lem: cat for cat, lems in SPEECH_LEXICON.items() for lem in lems}


def _lemma_as(word: str, pos: str) -> str:
    for p in morph.parse(word):
        if p.tag.POS == pos:
            return p.normal_form          # GRND -> infinitive, ADVB -> itself
    return word.lower()


def manner_modifiers(verb_tok):
    """Only words syntactically governed by the speech verb."""
    adv, conv = [], []
    for ch in verb_tok.children:
        if ch.pos_ == "ADV" and ch.dep_ == "advmod":
            adv.append(_lemma_as(ch.text, "ADVB"))
        elif "Conv" in ch.morph.get("VerbForm"):
            conv.append(_lemma_as(ch.text, "GRND"))
    return adv, conv


def analyse(doc, row):
    verbs = [(t, best_verb_parse(t.text, t.pos_)) for t in doc if t.is_alpha]
    finite = [(t, p) for t, p in verbs if p is not None and p.tag.POS == "VERB"]
    if not finite:
        return None
    # post: first finite verb after the dash; pre: finite verb closest to the colon
    tok, fp = finite[0] if row["position"] == "post" else finite[-1]
    lemma = fp.normal_form
    if lemma in PHASE_VERBS:                                   # "поспешил пробормотать"
        inf = next((p for t, p in verbs if p is not None and p.tag.POS == "INFN"
                    and 0 < t.i - tok.i <= 4), None)
        if inf:
            lemma = inf.normal_form
    gender, src = morph_gender(fp), "morph"
    if gender is None:
        gender, src = subject_gender(doc, tok, row["names"]), "subject"
    adv, conv = manner_modifiers(tok)
    return dict(lemma=lemma, form=tok.text.lower(), finite_lemma=fp.normal_form,
                aspect=_g(fp.tag.aspect), gender=gender or "unk", gender_source=src,
                inquit_lemmas=" ".join(morph.parse(t.text)[0].normal_form
                                       for t in doc if t.is_alpha),
                adverbs=" ".join(adv), gerunds=" ".join(conv))


def sync_categories(df, path: Path) -> dict:
    """Categories are assigned after extraction: new lemmas are appended to the CSV,
    pre-filled from SEED_CATEGORIES as suggestions; review and edit by hand."""
    freq = df.lemma.value_counts()
    cat = (pd.read_csv(path, dtype=str).fillna("") if path.exists()
           else pd.DataFrame(columns=["lemma", "category"]))
    new = sorted(set(freq.index) - set(cat.lemma))
    if new:
        cat = pd.concat([cat, pd.DataFrame({"lemma": new,
                         "category": [SEED_CATEGORIES.get(l, "") for l in new]})])
        logging.warning("%d new lemmas added to %s — please review", len(new), path)
    cat["freq"] = cat.lemma.map(freq).fillna(0).astype(int)
    cat.sort_values("freq", ascending=False).to_csv(path, index=False)
    lab = cat[cat.category != ""]
    return dict(zip(lab.lemma, lab.category))


# ======================================================= 2. comparison
def chapter_bootstrap(mf, lemmas, n_boot, rng):
    """95% CI of the per-mille difference (femn − masc), resampling whole chapters."""
    units = sorted(mf.unit.unique())
    def mat(g):
        sub = mf[mf.gender == g]
        return (pd.crosstab(sub.unit, sub.lemma)
                .reindex(index=units, columns=lemmas, fill_value=0).to_numpy(float))
    cm, cf = mat("masc"), mat("femn")
    W = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=n_boot)
    bm, bf = W @ cm, W @ cf                                   # (n_boot × lemmas), cheap
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = bf / bf.sum(1, keepdims=True) * 1000 - bm / bm.sum(1, keepdims=True) * 1000
    return np.nanpercentile(diff, [2.5, 97.5], axis=0)


def rarefied_overlap(mf, reps, rng):
    """Shrink the larger gender to the size of the smaller one, then compare type sets."""
    sizes = mf.gender.value_counts()
    big, small = sizes.idxmax(), sizes.idxmin()
    pool = mf.loc[mf.gender == big, "lemma"].to_numpy()
    s_set = set(mf.loc[mf.gender == small, "lemma"])
    res = []
    for _ in range(reps):
        b = set(rng.choice(pool, sizes[small], replace=False))
        res.append((len(b), len(b & s_set), len(b & s_set) / len(b | s_set)))
    return big, dict(zip(["types", "shared", "jaccard"], np.mean(res, axis=0)))


def k_for_coverage(counts, q):
    c = np.sort(counts[counts > 0])[::-1].cumsum() / counts.sum()
    return int(np.searchsorted(c, q) + 1)


def compare(df, out: Path, min_freq=5, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    mf = df[df.gender.isin(["masc", "femn"])].copy()
    mf["unit"] = mf.part.astype(str) + "." + mf.chapter.astype(str)
    ct = pd.crosstab(mf.lemma, mf.gender).reindex(columns=["masc", "femn"], fill_value=0)
    N = ct.sum()

    t = ct.copy()
    t["total"] = t.masc + t.femn
    t["pm_masc"], t["pm_femn"] = t.masc / N.masc * 1000, t.femn / N.femn * 1000
    t["diff_pm"] = t.pm_femn - t.pm_masc                      # >0 = more typical of femn
    t["log2_ratio"] = np.log2(((t.femn + .5) / N.femn) / ((t.masc + .5) / N.masc))
    for g in ("masc", "femn"):
        t[f"rank_{g}"] = t[g].where(t[g] > 0).rank(ascending=False, method="min")
    t["status"] = np.select([(t.masc > 0) & (t.femn > 0), t.masc > 0],
                            ["shared", "only_masc"], "only_femn")
    t["diff_lo"], t["diff_hi"] = chapter_bootstrap(mf, t.index, n_boot, rng)
    big = t.total >= min_freq
    if big.any():
        p = [fisher_exact([[r.masc, N.masc - r.masc], [r.femn, N.femn - r.femn]])[1]
             for r in t[big].itertuples()]
        t.loc[big, "p_fisher"] = p
        t.loc[big, "q_bh"] = multipletests(p, method="fdr_bh")[1]
    t = t.sort_values("diff_pm")
    t.to_csv(out / "verb_table.csv")

    with open(out / "sorted_lists.txt", "w", encoding="utf8") as fh:
        for status, col in (("shared", "total"), ("only_masc", "masc"), ("only_femn", "femn")):
            s = t[t.status == status].sort_index().sort_values(col, ascending=False, kind="stable")
            fh.write(f"=== {status}: {len(s)} types ===\n")
            fh.writelines(f"{l}\tm={r.masc}\tf={r.femn}\t{r.pm_masc:.1f}‰ / {r.pm_femn:.1f}‰\n"
                          for l, r in s.iterrows())
            fh.write("\n")

    sh = t[t.status == "shared"]
    rho, p_rho = spearmanr(sh.masc, sh.femn) if len(sh) > 2 else (np.nan, np.nan)
    big_g, rare = rarefied_overlap(mf, 200, rng)
    asp = pd.crosstab(mf.gender, mf.aspect.eq("perf")).reindex(
        index=["masc", "femn"], columns=[True, False], fill_value=0)
    summary = {
        "tokens_masc": N.masc, "tokens_femn": N.femn,
        "types_masc": int((t.masc > 0).sum()), "types_femn": int((t.femn > 0).sum()),
        "hapax_masc": int((t.masc == 1).sum()), "hapax_femn": int((t.femn == 1).sum()),
        "shared": len(sh), "only_masc": int((t.status == "only_masc").sum()),
        "only_femn": int((t.status == "only_femn").sum()),
        f"rarefied_{big_g}_types": rare["types"], "rarefied_shared": rare["shared"],
        "rarefied_jaccard": rare["jaccard"],
        "k50_masc": k_for_coverage(t.masc.values, .5), "k50_femn": k_for_coverage(t.femn.values, .5),
        "k80_masc": k_for_coverage(t.masc.values, .8), "k80_femn": k_for_coverage(t.femn.values, .8),
        "spearman_shared": rho, "spearman_p": p_rho,
        "perf_share_masc": asp.loc["masc", True] / asp.loc["masc"].sum(),
        "perf_share_femn": asp.loc["femn", True] / asp.loc["femn"].sum(),
        "perf_fisher_p": fisher_exact(asp.to_numpy())[1],
        "n_significant_q05": int((t.q_bh < .05).sum()) if "q_bh" in t else 0,
        "gender_from_subject_share": (mf.gender_source == "subject").mean(),
    }
    if "нейтральные" in set(mf.category):
        for g in ("masc", "femn"):
            summary[f"neutral_share_{g}"] = (mf[mf.gender == g].category == "нейтральные").mean()
    pd.Series(summary).to_csv(out / "summary.csv", header=["value"])
    logging.info("Summary:\n%s", pd.Series(summary).to_string())
    return t, mf


# ======================================================= 3. plots
COL = {"masc": "#3b6ea8", "femn": "#c0504d"}


def plot_scatter(t, out, min_freq, seed=0):
    rng = np.random.default_rng(seed)
    sh = t[t.status == "shared"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 8.5),
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    jit = lambda v: v * np.exp(rng.normal(0, .04, len(v)))   # separates identical points
    x, y = jit(sh.pm_masc.values), jit(sh.pm_femn.values)
    ax.scatter(x, y, s=8 * np.sqrt(sh.total), alpha=.6)
    lim = [min(x.min(), y.min()) * .8, max(x.max(), y.max()) * 1.2]
    ax.plot(lim, lim, "k--", lw=.8)
    texts = [ax.text(xi, yi, lem, fontsize=8)
             for xi, yi, lem, n in zip(x, y, sh.index, sh.total) if n >= min_freq]
    try:
        from adjustText import adjust_text                  # pip install adjustText
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=.4, color="grey"))
    except ImportError:
        pass
    ax.set(xscale="log", yscale="log", xlabel="masc, ‰", ylabel="femn, ‰",
           title=f"Shared verbs ({len(sh)}); above the diagonal — more frequent in femn remarks")

    om = t[t.status == "only_masc"].masc.nlargest(12)
    of = t[t.status == "only_femn"].femn.nlargest(12)
    ex = pd.concat([-om.sort_values(ascending=False), of.sort_values()])
    ax2.barh(ex.index, ex.values, color=[COL["masc"] if v < 0 else COL["femn"] for v in ex])
    ax2.axvline(0, color="k", lw=.6)
    ax2.set(xlabel="abs frequency   ← excl. masc | excl. femn →",
            title=f"Exclusive verbs (masc: {(t.status == 'only_masc').sum()}, "
                  f"femn: {(t.status == 'only_femn').sum()} типов)")
    fig.tight_layout(); fig.savefig(out / "02_scatter_exclusive.png", dpi=200); plt.close(fig)


def plot_diff(t, out, min_freq):
    d = t[t.total >= min_freq].sort_values("diff_pm")
    sig = (d.q_bh < .05).to_numpy() if "q_bh" in d else np.zeros(len(d), bool)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, .28 * len(d) + 1.5))
    ax.errorbar(d.diff_pm, y, fmt="none", ecolor="grey", lw=1,
                xerr=[np.clip(d.diff_pm - d.diff_lo, 0, None),
                      np.clip(d.diff_hi - d.diff_pm, 0, None)])
    ax.scatter(d.diff_pm, y, zorder=3, s=np.where(sig, 60, 20),
               c=[COL["femn"] if v > 0 else COL["masc"] for v in d.diff_pm],
               edgecolor="k", linewidths=np.where(sig, .8, 0))
    ax.set_yticks(y, d.index); ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="diff, per 1000 remarks (femn − masc); 95% bootstrap CI per chapter",
           title=f"Verbs with frequency ≥ {min_freq}; larger data points — q < 0.05 (Fisher + BH)")
    fig.tight_layout(); fig.savefig(out / "03_difference.png", dpi=200); plt.close(fig)

def plot_rest(df, t, mf, out, top=20):
    """Former plots 01, 04 (only if categories exist), 05, 06."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    fig, axes = plt.subplots(1, 2, figsize=(14, .35 * top + 2))
    for ax, g in zip(axes, ("masc", "femn")):
        s = t[f"pm_{g}"][t[g] > 0].nlargest(top)
        sns.barplot(x=s.values, y=s.index, ax=ax, color=COL[g])
        ax.set(title=f"{g}: топ-{top} (n={t[g].sum()})", xlabel="per 1000 remarks", ylabel="")
    fig.tight_layout(); fig.savefig(out / "01_top_verbs.png", dpi=200); plt.close(fig)

    lab = mf[~mf.category.isin(["—", "не речь"])]
    if lab.category.nunique() > 1:
        tab = pd.crosstab(lab.category, lab.gender)
        chi2, p, _, exp = chi2_contingency(tab)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        sns.heatmap(tab / tab.sum() * 100, annot=True, fmt=".1f", cmap="Blues", ax=axes[0])
        sns.heatmap((tab - exp) / np.sqrt(exp), annot=True, fmt=".1f", cmap="coolwarm",
                    center=0, ax=axes[1])
        axes[0].set_title("Category share, %")
        axes[1].set_title(f"Rest (χ²={chi2:.1f}, p={p:.3g})")
        fig.tight_layout(); fig.savefig(out / "04_categories.png", dpi=200); plt.close(fig)

    rng = np.random.default_rng(42)
    nmax = int(mf.gender.value_counts().min())
    sizes = np.unique(np.linspace(10, nmax, 25).astype(int))
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in ("masc", "femn"):
        pool = mf.loc[mf.gender == g, "lemma"].to_numpy()
        ax.plot(sizes, [np.mean([len(set(rng.choice(pool, s, replace=False)))
                                 for _ in range(100)]) for s in sizes], label=g, color=COL[g])
    ax.set(xlabel="sample size", ylabel="type count", title="Lexicon richness")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "05_rarefaction.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="gender", hue="layout", ax=axes[0],
                  order=["masc", "femn", "neut", "plur", "unk"])
    (mf.groupby("part").gender.value_counts(normalize=True).unstack()
       .reindex(columns=["masc", "femn"])
       .plot(marker="o", ax=axes[1], color=[COL["masc"], COL["femn"]]))
    fig.tight_layout(); fig.savefig(out / "06_confounds.png", dpi=200); plt.close(fig)

# ======================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml")
    ap.add_argument("-o", "--outdir", default="results")
    ap.add_argument("--spacy-model", default="ru_core_news_lg")
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--categories", default="/inquit/verb_categories.csv",
                    help="lemma→category CSV, created/updated automatically")
    ap.add_argument("--speech-only", action="store_true",
                    help="drop lemmas labelled 'не речь' or unlabelled in the categories CSV")
    ap.add_argument("--dash-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    cands = extract_candidates(etree.parse(args.xml).getroot())
    nlp = spacy.load(args.spacy_model, exclude=["ner"])
    rows = [{**c, **r, "names": "; ".join(c["names"])}
            for c, doc in zip(cands, nlp.pipe((c["inquit"] for c in cands), batch_size=256))
            if (r := analyse(doc, c))]
    df = pd.DataFrame(rows)

    cats = sync_categories(df, Path(args.categories))
    df["category"] = df.lemma.map(cats).fillna("—")
    if args.speech_only:
        df = df[~df.category.isin(["—", "не речь"])]
    if args.dash_only:
        df = df[df.layout == "dash"]
    df.to_csv(out / "inquits.csv", index=False)

    t, mf = compare(df, out, args.min_freq)
    plot_rest(df, t, mf, out)
    plot_scatter(t, out, args.min_freq)
    plot_diff(t, out, args.min_freq)
    logging.info("Done: %s", out.resolve())

if __name__ == "__main__":
    main()
