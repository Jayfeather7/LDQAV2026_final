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
                    "промолвить", "вымолвить", "молвить", "отозваться", "выговорить"},
    "вопрос": {"спросить", "спрашивать", "переспросить", "осведомиться", "допрашивать"},
    "ответ/реакция": {"ответить", "отвечать", "возразить", "возражать", "подтвердить",
                      "согласиться", "отрезать", "огрызнуться"},
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
LEMMA2CAT = {lem: cat for cat, lems in SPEECH_LEXICON.items() for lem in lems}
NEUTRAL = SPEECH_LEXICON["нейтральные"]

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
    for div1 in root.iter(TEI + "div1"):
        for div3 in div1.iter(TEI + "div3"):
            for p in div3.iter(P):
                yield div1.get("n"), div3.get("n"), p


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


def morph_gender(p):
    if p.tag.POS == "VERB" and p.tag.tense == "past":
        return "plur" if p.tag.number == "plur" else p.tag.gender
    return None


@lru_cache(maxsize=None)
def name_gender(name: str):
    for tok in name.split():
        if tok in NAME_GENDER_OVERRIDES:
            return NAME_GENDER_OVERRIDES[tok]
        for p in morph.parse(tok):
            if p.tag.grammemes & {"Name", "Surn", "Patr"} and p.tag.gender:
                return p.tag.gender
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
                return p.tag.gender
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


def analyse(doc, row):
    verbs = [(t, best_verb_parse(t.text, t.pos_)) for t in doc if t.is_alpha]
    verbs = [(t, p) for t, p in verbs if p is not None]
    if row["position"] == "post":
        finite = next(((t, p) for t, p in verbs if p.tag.POS == "VERB"), None)
        if finite is None:
            return None
        tok, fp = finite
        lemma = fp.normal_form
        if lemma in PHASE_VERBS:                       # "поспешил пробормотать"
            inf = next(((t, p) for t, p in verbs
                        if p.tag.POS == "INFN" and 0 < t.i - tok.i <= 4), None)
            if inf:
                lemma = inf[1].normal_form
    else:                                              # pre-inquit: last lexicon verb
        cand = [(t, p) for t, p in verbs
                if p.tag.POS == "VERB" and p.normal_form in LEMMA2CAT]
        if not cand:
            return None
        tok, fp = cand[-1]
        lemma = fp.normal_form

    gender, src = morph_gender(fp), "morph"
    if gender is None:
        gender, src = subject_gender(doc, tok, row["names"]), "subject"
    return dict(
        lemma=lemma, form=tok.text.lower(), finite_lemma=fp.normal_form,
        aspect=fp.tag.aspect, gender=gender or "unk", gender_source=src,
        category=LEMMA2CAT.get(lemma, "прочие"),
        inquit_lemmas=" ".join(morph.parse(t.text)[0].normal_form
                               for t in doc if t.is_alpha),
        adverbs=" ".join(morph.parse(t.text)[0].normal_form
                         for t in doc if t.pos_ == "ADV"),
    )


# ------------------------------------------------------------------ comparison
def log_odds_dirichlet(c1, c2, a0=100.0):
    """Monroe, Colaresi & Quinn (2008) log-odds with informative Dirichlet prior -> z."""
    prior = (c1 + c2) / (c1 + c2).sum() * a0
    n1, n2 = c1.sum(), c2.sum()
    d = (np.log((c1 + prior) / (n1 + a0 - c1 - prior))
         - np.log((c2 + prior) / (n2 + a0 - c2 - prior)))
    return d / np.sqrt(1 / (c1 + prior) + 1 / (c2 + prior))


def compare(df, out: Path, min_freq=5):
    mf = df[df.gender.isin(["masc", "femn"])]
    ct = mf.pivot_table(index="lemma", columns="gender", aggfunc="size", fill_value=0)
    ct = ct.reindex(columns=["masc", "femn"], fill_value=0)
    N = ct.sum()
    rel = ct / N * 1000                               # per 1000 inquits of that gender

    # sorted lists (frequency desc, alphabetical within ties)
    with open(out / "sorted_lists.txt", "w", encoding="utf8") as fh:
        for g in ("masc", "femn"):
            s = ct.loc[ct[g] > 0, g].sort_index().sort_values(ascending=False, kind="stable")
            fh.write(f"=== {g}: {s.sum()} tokens, {len(s)} types ===\n")
            fh.writelines(f"{lem}\t{n}\t{rel.at[lem, g]:.1f}‰\n" for lem, n in s.items())
            fh.write(f"\n--- {g}, alphabetical ---\n" + ", ".join(sorted(s.index)) + "\n\n")

    m_set, f_set = set(ct.index[ct.masc > 0]), set(ct.index[ct.femn > 0])
    shared = m_set & f_set
    rho, p_rho = spearmanr(ct.loc[list(shared), "masc"], ct.loc[list(shared), "femn"]) \
        if len(shared) > 2 else (np.nan, np.nan)

    stats = ct.copy()
    stats["rel_masc"], stats["rel_femn"] = rel.masc, rel.femn
    stats["z_logodds"] = log_odds_dirichlet(ct.masc.values.astype(float),
                                            ct.femn.values.astype(float))
    big = stats[(stats.masc + stats.femn) >= min_freq].copy()
    res = [fisher_exact([[r.masc, N.masc - r.masc], [r.femn, N.femn - r.femn]])
           for r in big.itertuples()]
    big["odds_ratio"] = [r[0] for r in res]
    big["p"] = [r[1] for r in res]
    big["p_bh"] = multipletests(big.p, method="fdr_bh")[1] if len(big) else []
    stats = stats.join(big[["odds_ratio", "p", "p_bh"]])
    stats.sort_values("z_logodds").to_csv(out / "verb_stats.csv")

    summary = {
        "tokens_masc": int(N.masc), "tokens_femn": int(N.femn),
        "types_masc": len(m_set), "types_femn": len(f_set),
        "shared": len(shared), "only_masc": len(m_set - f_set),
        "only_femn": len(f_set - m_set),
        "jaccard": len(shared) / max(1, len(m_set | f_set)),
        "spearman_shared": rho, "spearman_p": p_rho,
        "neutral_share_masc": ct.loc[ct.index.isin(NEUTRAL), "masc"].sum() / N.masc,
        "neutral_share_femn": ct.loc[ct.index.isin(NEUTRAL), "femn"].sum() / N.femn,
        "perfective_share_masc": (mf[mf.gender == "masc"].aspect == "perf").mean(),
        "perfective_share_femn": (mf[mf.gender == "femn"].aspect == "perf").mean(),
    }
    pd.Series(summary).to_csv(out / "summary.csv", header=["value"])
    logging.info("Summary:\n%s", pd.Series(summary).to_string())
    return ct, rel, stats, mf


# ------------------------------------------------------------------ plots
COL = {"masc": "#3b6ea8", "femn": "#c0504d"}


def plot_all(df, ct, rel, stats, mf, out: Path, top=20):
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # 1. top-N per gender
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.35 * top + 2))
    for ax, g in zip(axes, ("masc", "femn")):
        s = rel[g][rel[g] > 0].sort_values(ascending=False).head(top)
        sns.barplot(x=s.values, y=s.index, ax=ax, color=COL[g])
        ax.set(title=f"{g}: топ-{top} глаголов ремарок (n={ct[g].sum()})",
               xlabel="на 1000 ремарок", ylabel="")
    fig.tight_layout(); fig.savefig(out / "01_top_verbs.png", dpi=200); plt.close(fig)

    # 2. scatter masc vs femn (pseudocount 0.5 for log scale)
    N = ct.sum()
    x, y = (ct.masc + .5) / N.masc * 1000, (ct.femn + .5) / N.femn * 1000
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, s=10 + 3 * np.sqrt(ct.sum(axis=1)), alpha=.6)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "k--", lw=.8)
    for lem in (ct.masc + ct.femn).sort_values(ascending=False).head(30).index:
        ax.annotate(lem, (x[lem], y[lem]), fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="masc, ‰", ylabel="femn, ‰",
           title="Относительная частотность: выше диагонали — «женские» глаголы")
    fig.tight_layout(); fig.savefig(out / "02_scatter.png", dpi=200); plt.close(fig)

    # 3. diverging log-odds
    z = stats.z_logodds.sort_values()
    sel = pd.concat([z.head(15), z.tail(15)])
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(sel.index, sel.values, color=[COL["femn"] if v < 0 else COL["masc"] for v in sel])
    for v in (-1.96, 1.96):
        ax.axvline(v, color="grey", ls=":")
    ax.set(xlabel="z (log-odds, Dirichlet prior)  ←femn | masc→",
           title="Глаголы, маркированные по роду")
    fig.tight_layout(); fig.savefig(out / "03_logodds.png", dpi=200); plt.close(fig)

    # 4. semantic categories × gender: shares + standardized residuals
    tab = pd.crosstab(mf.category, mf.gender)
    chi2, p, _, exp = chi2_contingency(tab)
    resid = (tab - exp) / np.sqrt(exp)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sns.heatmap(tab / tab.sum() * 100, annot=True, fmt=".1f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Доля категории в ремарках рода, %")
    sns.heatmap(resid, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title(f"Стандартизованные остатки (χ²={chi2:.1f}, p={p:.3g})")
    fig.tight_layout(); fig.savefig(out / "04_categories.png", dpi=200); plt.close(fig)

    # 5. rarefaction: lexical diversity at equal sample sizes
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 5))
    nmax = int(min(ct.sum()))
    sizes = np.unique(np.linspace(10, nmax, 25).astype(int))
    for g in ("masc", "femn"):
        pool = mf.loc[mf.gender == g, "lemma"].to_numpy()
        ax.plot(sizes, [np.mean([len(set(rng.choice(pool, s, replace=False)))
                                 for _ in range(100)]) for s in sizes],
                label=g, color=COL[g])
    ax.set(xlabel="размер выборки (ремарок)", ylabel="число разных глаголов",
           title="Кривая разрежения (лексическое разнообразие)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "05_rarefaction.png", dpi=200)
    plt.close(fig)

    # 6. confounds: layout («» vs dialogue dash) and novel parts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="gender", hue="layout", ax=axes[0],
                  order=["masc", "femn", "neut"])
    axes[0].set_title("Род × оформление реплики")
    share = (mf.groupby("part").gender.value_counts(normalize=True)
             .unstack().reindex(columns=["masc", "femn"]))
    share.plot(marker="o", ax=axes[1], color=[COL["masc"], COL["femn"]])
    axes[1].set(title="Доля родов по частям романа", xlabel="часть", ylabel="доля")
    fig.tight_layout(); fig.savefig(out / "06_confounds.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml")
    ap.add_argument("-o", "--outdir", default="results")
    ap.add_argument("--spacy-model", default="ru_core_news_lg")
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--speech-only", action="store_true",
                    help="keep only verbs from SPEECH_LEXICON")
    ap.add_argument("--dash-only", action="store_true",
                    help="exclude «»-framed (mostly inner/embedded) speech")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    root = etree.parse(args.xml).getroot()
    cands = extract_candidates(root)
    logging.info("Candidate inquits: %d", len(cands))

    nlp = spacy.load(args.spacy_model, exclude=["ner"])
    rows = []
    for row, doc in zip(cands, nlp.pipe((c["inquit"] for c in cands), batch_size=256)):
        res = analyse(doc, row)
        if res:
            rows.append({**row, **res, "names": "; ".join(row["names"])})
    df = pd.DataFrame(rows)
    if args.speech_only:
        df = df[df.category != "прочие"]
    if args.dash_only:
        df = df[df.layout == "dash"]
    df.to_csv(out / "inquits.csv", index=False)

    ct, rel, stats, mf = compare(df, out, args.min_freq)
    plot_all(df, ct, rel, stats, mf, out)
    logging.info("Done. See %s", out.resolve())


if __name__ == "__main__":
    main()
