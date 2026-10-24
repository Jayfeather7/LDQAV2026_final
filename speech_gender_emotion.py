#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gendered speech verbs + NRC emotion in a Digital-Dostoevsky TEI novel.
python speech_gender_emotion.py --input prest_i_nakaz_tagged.txt --characters characters.csv \
       --nrc NRC-multilingual.xlsx --out results
"""
import argparse, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, spacy
from lxml import etree
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMO = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
CATS = EMO + ["positive", "negative"]
CIDX = {c: i for i, c in enumerate(CATS)}
SEED = """сказать говорить спросить спрашивать ответить отвечать воскликнуть восклицать крикнуть кричать
закричать вскричать вскрикнуть шепнуть шептать прошептать заметить возразить возражать прибавить добавить
пробормотать бормотать проговорить произнести промолвить заявить объявить повторить перебить прервать
подтвердить приказать взвизгнуть прохрипеть пролепетать пояснить объяснить продолжить подхватить
возопить заорать""".split()
ASPECT = {"говорить": "сказать", "отвечать": "ответить", "спрашивать": "спросить", "восклицать": "воскликнуть",
          "возражать": "возразить", "кричать": "крикнуть", "бормотать": "пробормотать", "шептать": "шепнуть",
          "лепетать": "пролепетать", "добавлять": "добавить", "прибавлять": "прибавить"}
PHASAL = {"поспешить", "стать", "начать", "приняться", "успеть", "решиться", "осмелиться", "сметь",
          "пытаться", "попытаться"}   # "поспешил пробормотать" -> speech verb = infinitive
KEEP_DEPS = {"advmod", "obl", "obj", "iobj", "advcl", "xcomp", "ccomp", "parataxis", "nmod"}
COL = {"male": "#3b6fb6", "female": "#c8553d", "neutral": "#8a8a8a", "rare": "#cccccc", "M": "#3b6fb6", "F": "#c8553d"}
DASH = " \t—–-"
SENT = re.compile(r"(?<=[.!?…])\s+")

norm = lambda s: str(s).lower().replace("ё", "е").strip()
canon = lambda l, m=True: ASPECT.get(norm(l), norm(l)) if m else norm(l)
light = lambda nlp: [c for c in ("parser", "ner") if c in nlp.pipe_names]
first_sent = lambda t: SENT.split(re.sub(r"^[\s,;:—–-]+", "", t or ""), 1)[0].strip(DASH)
last_sent = lambda t: SENT.split((t or "").strip())[-1].strip(DASH)


# ------------------------------------------------------------------ TEI INPUT
def read_tei(path, keep_quoted):
    """-> utterances (top-level <said>), frames (narration adjacent to speech), background narration."""
    root = etree.parse(str(path)).getroot()
    utts, frames, bg, carry = [], [], [], ""
    for p in root.iter("{*}p"):
        kids = [c for c in p if isinstance(c.tag, str)]
        said = [c for c in kids if etree.QName(c).localname == "said"]
        if not said:                                  # pure narration paragraph
            carry = "".join(p.itertext())
            if len(carry) > 40:
                bg.append(carry)
            continue
        base = len(utts)
        for s in said:
            t = "".join(s.itertext()).strip()
            utts.append(dict(id=len(utts), text=t, quoted=t.startswith(("«", '"', "„")),
                             who=next((s.get(a) for a in ("who", "addresser", "speaker") if s.get(a)), None)))
        for j, s in enumerate(said):                  # frame = start of the tail after </said>
            tail, ids, nx = s.tail or "", [base + j], s.getnext()
            if nx is not None and etree.QName(nx).localname == "said" and tail.rstrip().endswith(("—", "–", "-", ",", ":")):
                ids.append(base + j + 1)              # tail sits between two saids: same utterance continues
            if first_sent(tail):
                frames.append(dict(text=first_sent(tail), uids=ids))
        pre = p.text or ""                            # frame before the first said ("... воскликнул:")
        if not pre.strip(DASH) and carry.rstrip().endswith(":"):
            pre = carry
        if pre.rstrip().endswith(":") and last_sent(pre):
            frames.append(dict(text=last_sent(pre), uids=[base]))
        carry = (kids[-1].tail or "") if kids else ""
    if not keep_quoted:                               # drop thoughts / quotations («...»)
        frames = [f for f in frames if not utts[f["uids"][0]]["quoted"]]
    return utts, frames, bg


# ------------------------------------------------------------ GENDER ATTRIBUTION
def load_table(path):
    t = {}
    if path and Path(path).exists():
        for _, r in pd.read_csv(path, comment="#").iterrows():
            t[norm(r["alias"])] = (str(r["gender"]).strip().upper()[0], str(r["character"]))
    else:
        print("[warn] no character table: gender from agreement/pronouns only")
    return t


def subject_of(t):
    for c in t.children:
        if c.dep_.startswith("nsubj"):
            return c
    return subject_of(t.head) if t.dep_ == "conj" and t.head is not t else None


def agreement(v):
    m = v.morph
    if "Past" in m.get("Tense") and "Sing" in m.get("Number"):
        g = m.get("Gender")
        return "M" if "Masc" in g else "F" if "Fem" in g else None


def lookup(subj, table):
    for c in [subj] + [c for c in subj.children if c.dep_.split(":")[0] in ("flat", "appos", "nmod", "amod")]:
        for k in (norm(c.text), norm(c.lemma_)):
            if k in table:
                return table[k]


def resolve(src, who, table):
    """TEI @who -> past-tense agreement -> character table -> pronoun -> animate noun."""
    subj, agree = subject_of(src), agreement(src)
    hit = lookup(subj, table) if subj is not None else None
    g = meth = None
    char = hit[1] if hit else ""
    if who and norm(who).lstrip("#") in table:
        g, char = table[norm(who).lstrip("#")]; meth = "tei_who"
    elif agree:
        g, meth = agree, "agreement"
    elif hit:
        g, meth = hit[0], "table"
    elif subj is not None:
        w = {norm(subj.text), norm(subj.lemma_)}
        if subj.pos_ == "PRON" and w & {"он", "она"}:
            g, meth = ("M" if "он" in w else "F"), "pronoun"
        elif subj.pos_ == "PROPN" or (subj.pos_ == "NOUN" and "Anim" in subj.morph.get("Animacy")):
            mg = subj.morph.get("Gender")
            g = "M" if "Masc" in mg else "F" if "Fem" in mg else None
            meth = "noun" if g else None
    return g, meth, subj, agree, bool(hit and agree and hit[0] != agree), char


# ------------------------------------------------------------------ EXTRACTION
def heads(doc):
    """(speech_verb, governing_verb) for every clause-head verb of a frame."""
    out = []
    for t in doc:
        if t.pos_ == "VERB" and (t.dep_ == "ROOT" or (t.dep_ == "conj" and t.head.dep_ == "ROOT")):
            x = [c for c in t.children if c.dep_ == "xcomp" and c.pos_ == "VERB"]
            out.append((x[0] if x and norm(t.lemma_) in PHASAL else t, t))
    if not out:
        fin = [t for t in doc if t.pos_ == "VERB" and "Fin" in t.morph.get("VerbForm")]
        out = [(t, t) for t in fin[:1]]
    return out


def extract(frames, utts, bg, nlp, table, merge):
    rows = []
    for f, doc in zip(frames, nlp.pipe([f["text"] for f in frames], batch_size=64)):
        who = next((utts[i]["who"] for i in f["uids"] if utts[i]["who"]), None)
        for verb, src in heads(doc):
            g, meth, subj, agree, conf, char = resolve(src, who, table)
            rows.append(dict(
                verb=canon(verb.lemma_, merge), form=norm(verb.text), gender=g or "", method=meth or "unresolved",
                character=char, subj=subj.text if subj is not None else "",
                subj_lemma=norm(subj.lemma_) if subj is not None else "",
                subj_pos=subj.pos_ if subj is not None else "", agree=agree or "", conflict=conf,
                deps=";".join(f"{c.dep_.split(':')[0]}:{norm(c.lemma_)}" for c in verb.children
                              if c.dep_.split(":")[0] in KEEP_DEPS and c.pos_ != "PUNCT"),
                pos=f["uids"][0] / len(utts), frame=f["text"], uids=";".join(map(str, f["uids"]))))
    bgc = Counter(canon(t.lemma_, merge) for d in nlp.pipe(bg, disable=light(nlp), batch_size=32)
                  for t in d if t.pos_ == "VERB")
    return pd.DataFrame(rows), bgc


def candidates(occ, bgc, seed):
    c = occ.groupby("verb").size().rename("n_tag").to_frame()
    c["n_elsewhere"] = [bgc.get(v, 0) for v in c.index]
    c["tag_rate"] = c.n_tag / (c.n_tag + c.n_elsewhere)
    c["in_seed"] = c.index.isin(seed)
    c["example"] = occ.groupby("verb")["frame"].first()
    return c.sort_values("n_tag", ascending=False).reset_index()


def select_verbs(cands, a, merge):
    seed = {canon(v, merge) for v in SEED}
    if a.verbs == "seed":
        return seed
    if a.verbs == "file":
        return {canon(l, merge) for l in Path(a.verb_file).read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}
    return seed | set(cands[(cands.n_tag >= a.auto_min_tag) & (cands.tag_rate >= a.auto_min_rate)].verb)


# --------------------------------------------------------------- VERB STATISTICS
def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(r, 1)
    return out


def verb_stats(occ, lex, a):
    res = occ[occ.gender.isin(["M", "F"])]
    if res.empty:
        sys.exit("No occurrences with resolved gender - check characters.csv / model.")
    tab = res.groupby(["verb", "gender"]).size().unstack(fill_value=0).reindex(columns=["M", "F"], fill_value=0)
    totM, totF = int(tab.M.sum()), int(tab.F.sum())
    base = totF / (totM + totF)
    rows = []
    for v, r in tab.iterrows():
        m, f = int(r.M), int(r.F)
        lor = np.log(((f + .5) / (m + .5)) / ((totF - f + .5) / (totM - m + .5)))
        p = stats.fisher_exact([[f, m], [totF - f, totM - m]])[1]
        rows.append(dict(verb=v, m=m, f=f, n=m + f, share_f=f / (m + f), lor=lor, p=p,
                         nrc=",".join(sorted(lex.get(v, [])))))
    vs = pd.DataFrame(rows)
    vs["q"] = np.nan
    ok = vs.n >= a.min_n
    vs.loc[ok, "q"] = bh(vs.loc[ok, "p"])

    def cls(r):
        if r.n < a.min_n:
            return "rare"
        d = r.share_f - base
        if d > a.margin and r.lor >= a.lor_thr and r.q < a.alpha:
            return "female"
        if d < -a.margin and r.lor <= -a.lor_thr and r.q < a.alpha:
            return "male"
        return "neutral"
    vs["class"] = vs.apply(cls, axis=1)
    return vs.sort_values("n", ascending=False).reset_index(drop=True), base, totM, totF


# ------------------------------------------------------------------- NRC LEXICON
def load_nrc(path, nlp, lang):
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, sep=None, engine="python")
    cols = {str(c).strip().lower(): c for c in df.columns}
    raw = defaultdict(set)
    if set(CATS) <= set(cols):                                   # wide multilingual file
        lc = next((o for l, o in cols.items() if lang in l), None)
        if lc is None:
            sys.exit(f"No column containing '{lang}' in {list(df.columns)[:12]}...")
        num = df[[cols[c] for c in CATS]].apply(pd.to_numeric, errors="coerce").fillna(0).values
        for w, row in zip(df[lc], num):
            if isinstance(w, str) and w.strip():
                raw[w.strip()] |= {c for c, v in zip(CATS, row) if v > 0}
    elif {"word", "sentiment"} <= set(cols):                     # long format (e.g. exported from syuzhet)
        for _, r in df.iterrows():
            s = str(r[cols["sentiment"]]).lower()
            if s in CIDX and ("value" not in cols or pd.to_numeric(r[cols["value"]], errors="coerce") > 0):
                raw[str(r[cols["word"]]).strip()].add(s)
    else:
        sys.exit("Unrecognised NRC layout (need 10 category columns, or word+sentiment).")
    words = [w for w in raw if len(w.split()) == 1]
    lex = defaultdict(set)
    for w, d in zip(words, nlp.pipe(words, disable=light(nlp), batch_size=256)):
        for k in {norm(w)} | ({norm(d[0].lemma_)} if len(d) == 1 else set()):
            lex[k] |= raw[w]
    print(f"[nrc] {len(words)} single-word entries -> {len(lex)} lookup keys")
    return lex


def score_docs(docs, lex):
    """rows: 10 NRC counts, token count, tokens with >=1 of the 8 emotions."""
    out = []
    for d in docs:
        v = np.zeros(len(CATS) + 2)
        for t in d:
            if t.is_alpha:
                v[-2] += 1
                cats = lex.get(norm(t.lemma_)) or lex.get(norm(t.text))
                if cats:
                    for c in cats:
                        v[CIDX[c]] += 1
                    v[-1] += bool(cats & set(EMO))
        out.append(v)
    return np.array(out)


# --------------------------------------------------------------- EMOTION ANALYSIS
dens = lambda C, T: C.sum(0) / max(T.sum(), 1) * 1000


def emotion_analysis(occ, vs, utts, nlp, lex, a, rng, out):
    sp = occ[occ.gender.isin(["M", "F"])].copy()
    sp["ul"] = sp.uids.map(lambda s: [int(i) for i in s.split(";")])
    ids = sorted({i for l in sp.ul for i in l})
    sc = dict(zip(ids, score_docs(nlp.pipe([utts[i]["text"] for i in ids], disable=light(nlp), batch_size=64), lex)))
    ug = {}                                              # utterance -> speaker gender (AMBIG if contradictory)
    for ul, g in zip(sp.ul, sp.gender):
        for i in ul:
            ug[i] = g if ug.get(i, g) == g else "AMBIG"
    uid = [i for i in ids if ug[i] in ("M", "F")]
    U = np.array([sc[i] for i in uid])
    C, T = U[:, :len(CATS)], U[:, -2]
    isF = np.array([ug[i] == "F" for i in uid])
    print(f"[emotion] {len(uid)} utterances with unambiguous speaker gender (F={isF.sum()}, M={(~isF).sum()})")

    obs = dens(C[isF], T[isF]) - dens(C[~isF], T[~isF])               # permutation test (female - male)
    ge = np.zeros(len(CATS))
    for _ in range(a.n_perm):
        pm = rng.permutation(isF)
        ge += np.abs(dens(C[pm], T[pm]) - dens(C[~pm], T[~pm])) >= np.abs(obs)
    p = (ge + 1) / (a.n_perm + 1)

    def ci(mask):
        c, t = C[mask], T[mask]
        bs = np.array([dens(c[s], t[s]) for s in (rng.choice(len(c), len(c)) for _ in range(a.n_boot))])
        return np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    (loF, hiF), (loM, hiM) = ci(isF), ci(~isF)
    eg = pd.DataFrame(dict(category=CATS, per1000_F=dens(C[isF], T[isF]), per1000_M=dens(C[~isF], T[~isF]),
                           diff_F_minus_M=obs, p_perm=p, q_bh=bh(p), F_lo=loF, F_hi=hiF, M_lo=loM, M_hi=hiM))
    eg.to_csv(out / "emotion_by_gender.csv", index=False)
    utt = pd.DataFrame(dict(uid=uid, gender=[ug[i] for i in uid], tokens=T,
                            valence=C[:, CIDX["positive"]] - C[:, CIDX["negative"]]))
    utt.to_csv(out / "utterances.csv", index=False)

    S = np.array([sum(sc[i] for i in l) for l in sp.ul])             # emotion of speech each verb token introduces
    sp["tokens"], sp["emo_any"] = S[:, -2], S[:, -1]
    for c, i in CIDX.items():
        sp["emo_" + c] = S[:, i]
    sp["cls"] = sp.verb.map(vs.set_index("verb")["class"])
    sp.drop(columns="ul").to_csv(out / "occurrences_with_emotion.csv", index=False)
    byc = pd.DataFrame([dict(verb_class=k, n_occ=len(g), **{c: g["emo_" + c].sum() / max(g.tokens.sum(), 1) * 1000
                                                           for c in CATS}) for k, g in sp.groupby("cls")])
    byc.to_csv(out / "emotion_by_verb_class.csv", index=False)
    pv = sp.groupby("verb").agg(n=("verb", "size"), tokens=("tokens", "sum"), emo=("emo_any", "sum"))
    pv = pv[pv.n >= a.min_n].join(vs.set_index("verb")[["lor", "class"]])
    pv["emo_per1000"] = pv.emo / pv.tokens.clip(lower=1) * 1000
    rho = pr = np.nan
    if len(pv) >= 5:
        rho, pr = stats.spearmanr(pv.lor, pv.emo_per1000)
    pv.reset_index().to_csv(out / "verb_lor_vs_emotion.csv", index=False)
    return eg, utt, byc, pv, rho, pr


# ------------------------------------------------------------------------- PLOTS
def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, dpi=200); plt.close(fig)


def plot_all(occ, vs, base, eg, utt, byc, pv, rho, pr, a, out):
    d = vs[vs["class"] != "rare"].head(30).sort_values("lor")                      # fig1
    if len(d):
        fig, ax = plt.subplots(figsize=(8, .3 * len(d) + 1.5))
        ax.barh(d.verb, d.lor, color=d["class"].map(COL)); ax.axvline(0, color="k", lw=.8)
        for y, (l, n) in enumerate(zip(d.lor, d.n)):
            ax.text(l, y, f" n={n}", va="center", ha="left" if l >= 0 else "right", fontsize=7)
        ax.set_xlabel(f"log-odds (→ female); baseline female share = {base:.2f}")
        ax.set_title("Speech verbs by gender skew (blue male, red female, grey neutral)")
        save(fig, out, "fig1_verb_gender_skew.png")

    x = np.arange(len(eg)); fig, ax = plt.subplots(figsize=(10, 4.5))             # fig2
    for off, g, nm in ((-.2, "F", "female"), (.2, "M", "male")):
        y = eg[f"per1000_{g}"].values
        ax.bar(x + off, y, .4, color=COL[g], label=nm, capsize=2,
               yerr=np.vstack([y - eg[f"{g}_lo"].values, eg[f"{g}_hi"].values - y]))
    for i, q in enumerate(eg.q_bh):
        if q < .05:
            ax.text(i, max(eg.per1000_F[i], eg.per1000_M[i]) * 1.08, "*", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(eg.category, rotation=30); ax.legend()
    ax.set_ylabel("NRC words per 1000 tokens of direct speech")
    ax.set_title("Emotion in direct speech by speaker gender (95% bootstrap CI; * = BH q<0.05)")
    save(fig, out, "fig2_emotion_by_speaker_gender.png")

    if len(byc):                                                                   # fig3
        x = np.arange(len(CATS)); w = .8 / len(byc); fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, r in byc.reset_index(drop=True).iterrows():
            ax.bar(x + i * w - .4 + w / 2, [r[c] for c in CATS], w, color=COL.get(r.verb_class, "k"),
                   label=f"{r.verb_class} verbs (n={r.n_occ})")
        ax.set_xticks(x); ax.set_xticklabels(CATS, rotation=30); ax.legend()
        ax.set_ylabel("NRC words per 1000 tokens of introduced speech")
        ax.set_title("Emotion in speech introduced by male / female / neutral verbs")
        save(fig, out, "fig3_emotion_by_verb_class.png")

    if len(pv):                                                                    # fig4
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(pv.lor, pv.emo_per1000, s=pv.n * 3, c=pv["class"].map(COL), alpha=.7)
        for v, r in pv.iterrows():
            ax.annotate(v, (r.lor, r.emo_per1000), fontsize=7)
        ax.axvline(0, color="k", lw=.5); ax.set_xlabel("verb log-odds (→ female)")
        ax.set_ylabel("emotion-word density of introduced speech (per 1000 tokens)")
        ax.set_title(f"Gender skew vs emotionality (Spearman ρ={rho:.2f}, p={pr:.3f})")
        save(fig, out, "fig4_lor_vs_emotion.png")

    fig, ax = plt.subplots(figsize=(10, 4))                                        # fig5
    for g, nm in (("F", "female"), ("M", "male")):
        d = utt[utt.gender == g].sort_values("uid")
        if len(d) >= 10:
            w = min(a.arc_window, max(10, len(d) // 5))
            ax.plot(d.uid / utt.uid.max(), d.valence.rolling(w).sum() / d.tokens.rolling(w).sum().clip(lower=1) * 1000,
                    color=COL[g], label=f"{nm} (window={w} utt.)")
    ax.axhline(0, color="k", lw=.5); ax.legend(); ax.set_xlabel("position in novel (0-1)")
    ax.set_ylabel("net valence per 1000 tokens"); ax.set_title("Rolling valence of direct speech by speaker gender")
    save(fig, out, "fig5_valence_arc.png")

    el = vs[vs["class"].isin(["male", "female", "neutral"])]                       # fig6
    picks = [("male", v) for v in el[el["class"] == "male"].sort_values("lor").verb.head(3)]
    picks += [("female", v) for v in el[el["class"] == "female"].sort_values("lor", ascending=False).verb.head(3)]
    picks += [("neutral", v) for v in el[el["class"] == "neutral"].sort_values("n", ascending=False).verb.head(1)]
    if picks:
        nr = int(np.ceil(len(picks) / 4)); fig, axes = plt.subplots(nr, 4, figsize=(17, 3.2 * nr), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        res = occ[occ.gender.isin(["M", "F"])]
        for ax, (cl, v) in zip(axes.ravel(), picks):
            ax.axis("on")
            cnt = {g: Counter(d for s in res[(res.verb == v) & (res.gender == g)].deps.fillna("")
                              for d in s.split(";") if d) for g in "MF"}
            top = [d for d, _ in (cnt["M"] + cnt["F"]).most_common(6)][::-1]
            if top:
                ax.barh(top, [cnt["M"][d] for d in top], color=COL["M"], label="male")
                ax.barh(top, [cnt["F"][d] for d in top], left=[cnt["M"][d] for d in top], color=COL["F"], label="female")
            ax.set_title(f"{v} [{cl}]", fontsize=10, color=COL[cl]); ax.tick_params(labelsize=7)
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle("Syntactic dependents of top male / female / neutral verbs")
        save(fig, out, "fig6_verb_dependents.png")


# ------------------------------------------------------------- VALIDATION / REPORT
def validate(path):
    df = pd.read_csv(path)
    df = df[df.gold_gender.isin(["M", "F"])]
    if df.empty:
        sys.exit("Fill gold_gender (M/F) first.")
    df["ok"] = df.pred_gender == df.gold_gender
    print(f"Accuracy: {df.ok.mean():.3f} (n={len(df)})")
    print(df.groupby("method").ok.agg(["mean", "size"]))


def report(out, a, occ, vs, base, totM, totF, eg, rho, pr, cov, n_utt, n_frames):
    el = vs[vs.n >= a.min_n]
    L = [f"Utterances (top-level <said>): {n_utt}; frames: {n_frames}; speech-verb occurrences: {len(occ)}",
         f"  gender resolved: {totM + totF} (M={totM}, F={totF}); unresolved: {len(occ) - totM - totF}",
         f"  baseline female share: {base:.3f}", f"  attribution methods: {occ.method.value_counts().to_dict()}",
         f"  agreement-vs-table conflicts: {int(occ.conflict.sum())}",
         f"Verb classes (min_n={a.min_n}, margin={a.margin}, |LOR|>={a.lor_thr}, q<{a.alpha}): {vs['class'].value_counts().to_dict()}",
         f"NRC coverage of speech-verb lemmas: {cov:.1%}"]
    sig = False
    if len(el) >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(el[["m", "f"]].values)
        sig = p < a.alpha
        L.append(f"Global verb x gender chi-square: chi2={chi2:.1f}, dof={dof}, p={p:.4g}, "
                 f"Cramer's V={np.sqrt(chi2 / el[['m', 'f']].values.sum()):.3f}")
    ns = int(vs["class"].isin(["male", "female"]).sum())
    L.append(f"H0 (verb choice independent of speaker gender): {'REJECTED' if sig and ns else 'NOT rejected'} "
             f"at alpha={a.alpha} ({ns} skewed verbs)")
    s = eg[eg.q_bh < a.alpha]
    L.append(f"Emotion categories differing by speaker gender (BH q<{a.alpha}): {', '.join(s.category) or 'none'}")
    L.append(f"Spearman(verb LOR, emotionality of introduced speech): rho={rho:.3f}, p={pr:.4f}")
    (out / "report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ------------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input"); ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="ru_core_news_md"); ap.add_argument("--characters", default="characters.csv")
    ap.add_argument("--nrc"); ap.add_argument("--nrc-lang", default="russian")
    ap.add_argument("--keep-quoted", action="store_true", help="keep «...» utterances (thoughts, quotations)")
    ap.add_argument("--verbs", choices=["auto", "seed", "file"], default="auto"); ap.add_argument("--verb-file")
    ap.add_argument("--auto-min-tag", type=int, default=5); ap.add_argument("--auto-min-rate", type=float, default=.5)
    ap.add_argument("--no-merge-aspect", action="store_true")
    ap.add_argument("--min-n", type=int, default=10); ap.add_argument("--margin", type=float, default=.10)
    ap.add_argument("--lor-thr", type=float, default=.5); ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--n-perm", type=int, default=2000); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--arc-window", type=int, default=100); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", help="filled validation_sample.csv -> prints accuracy")
    a = ap.parse_args()
    if a.validate:
        return validate(a.validate)
    if not a.input:
        sys.exit("--input is required")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng, merge = np.random.default_rng(a.seed), not a.no_merge_aspect
    nlp, table = spacy.load(a.model), load_table(a.characters)

    utts, frames, bg = read_tei(a.input, a.keep_quoted)
    usable = [u for u in utts if a.keep_quoted or not u["quoted"]]
    cov = len({i for f in frames for i in f["uids"]})
    print(f"[tei] {len(utts)} utterances ({len(utts) - len(usable)} quoted/thought dropped), {len(frames)} frames, "
          f"{cov / max(len(usable), 1):.0%} of usable utterances have a frame, {len(bg)} pure-narration paragraphs")
    occ_all, bgc = extract(frames, utts, bg, nlp, table, merge)
    if occ_all.empty:
        sys.exit("No clause-head verbs found.")
    occ_all.to_csv(out / "occurrences_all_heads.csv", index=False)
    cands = candidates(occ_all, bgc, {canon(v, merge) for v in SEED})
    cands.to_csv(out / "verb_candidates.csv", index=False)
    chosen = select_verbs(cands, a, merge)
    occ = occ_all[occ_all.verb.isin(chosen)].reset_index(drop=True)
    print(f"[verbs] {len(chosen)} lemmas selected -> {len(occ)} occurrences")
    if occ.empty:
        sys.exit("No occurrences of the selected verbs.")
    occ.to_csv(out / "occurrences_speech_verbs.csv", index=False)
    (occ[occ.subj_pos.isin(["PROPN", "NOUN"]) & (occ.character == "")].groupby("subj_lemma")
        .agg(n=("verb", "size"), agree_M=("agree", lambda s: (s == "M").sum()), agree_F=("agree", lambda s: (s == "F").sum()))
        .sort_values("n", ascending=False).head(150).to_csv(out / "unmatched_subjects.csv"))
    sample = occ[occ.gender.isin(["M", "F"])].sample(min(200, int(occ.gender.isin(["M", "F"]).sum())), random_state=a.seed)
    pd.DataFrame(dict(occ_index=sample.index, frame=sample.frame,
                      speech=[" | ".join(utts[int(i)]["text"][:120] for i in s.split(";")) for s in sample.uids],
                      pred_gender=sample.gender, method=sample.method, gold_gender="")
                 ).to_csv(out / "validation_sample.csv", index=False)
    if not a.nrc:
        sys.exit("Gender/verb stages done (see results/). Add --nrc for the emotion analysis.")

    lex = load_nrc(a.nrc, nlp, a.nrc_lang)
    vs, base, totM, totF = verb_stats(occ, lex, a)
    vs.to_csv(out / "verb_stats.csv", index=False)
    eg, utt, byc, pv, rho, pr = emotion_analysis(occ, vs, utts, nlp, lex, a, rng, out)
    plot_all(occ, vs, base, eg, utt, byc, pv, rho, pr, a, out)
    report(out, a, occ, vs, base, totM, totF, eg, rho, pr, float(np.mean([v in lex for v in vs.verb])),
           len(utts), len(frames))


if __name__ == "__main__":
    main()
