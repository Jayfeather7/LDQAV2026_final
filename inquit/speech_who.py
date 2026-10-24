#!/usr/bin/env python3
"""
speech_who.py — speech verbs in inquits × speaker (@who) × addressee (@toWhom)
for the Digital Dostoevsky TEI of «Преступление и наказание».

Requires inquit_gender.py (version 2: analyse, sync_categories, compare,
plot_rest, plot_scatter, plot_diff, plot_manner) in the same folder.
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from difflib import get_close_matches
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from lxml import etree
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from inquit_gender2 import analyse, sync_categories

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"t": TEI_NS}
T = "{%s}" % TEI_NS
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
SAID, Q, P, PERS, SEG, DIV = (T + x for x in ("said", "q", "p", "persName", "seg", "div"))

DASHES = str.maketrans({"–": "—", "\u00a0": " "})
POST_RE = re.compile(r"^\s*[,.!?…;]*\s*—\s*(.+)", re.S)
CUT_RE = re.compile(r"[.!?…:;(]|\s—")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
SEX = {"male": "masc", "female": "femn"}
SEXCOL = {"masc": "#3b6ea8", "femn": "#c0504d", "other": "#999999"}
MODE_ORDER = ["реплика", "вслух себе", "про себя"]

# ------------------------------------------------------------------ annotation repair
ID_ALIASES = {"pnnats": "pnnast", "duk1": "dukl", "oepl": "pepl", "par": "palr",
              "ras": "rrr", "hoz": "hozbar", "titva": "titv", "$ssm": "ssm",
              "poepe": "porpe", "pole1": "pnpole1", "pnkon1": "pnkonv1", "afz": "agz"}

# (part, section, utterance-text prefix) -> corrected attributes;
# extend after reviewing conflicts_morph_vs_tei.csv
MANUAL_FIXES = {
    ("2", "7", "Эх! Расстроил я его"): {"who": "#dpr", "toWhom": "#dpr"},
    ("3", "5", "Значит, от убытка бог избавил"): {"who": "#dpr"},
    ("3", "5", "Ты, брат, кажется, надо мной подсмеиваешься"): {"toWhom": "#dpr"},
    ("3", "4", "Да вот этой, Софьи"): {"toWhom": "#arr"},
    ("4", "6", "Один убил?"): {"who": "#porpe", "toWhom": "#myde"},
    ("6", "5", "Нет, я не говорил ей ни слова"): {"toWhom": "#arr"},
    ("6", "5", "Господин Разумихин? Статью"): {"toWhom": "#arr"},
    ("6", "5", "Эге, Авдотья Романовна"): {"who": "#ais", "toWhom": "#arr"},
    ("6", "7", "маменька, оставьте это"): {"toWhom": "#palr"},
}

SHORT = {"rrr": "Раскольников", "ssm": "Соня", "arr": "Дуня", "palr": "Пульхерия Ал.",
         "dpr": "Разумихин", "porpe": "Порфирий", "ais": "Свидригайлов", "pepl": "Лужин",
         "kim": "Катерина Ив.", "szm": "Мармеладов", "pnnast": "Настасья",
         "zsm": "Зосимов", "agz": "Заметов", "ilpe": "Илья Петрович",
         "asl": "Лебезятников", "nikf": "Никодим Фомич", "alul": "Амалия Ив.",
         "star": "Алёна Ив.", "lizi": "Лизавета", "psm": "Поленька", "myde": "Миколка"}


def load_persons(root) -> dict:
    persons = {}
    for tag, grp in (("person", False), ("personGrp", True)):
        for el in root.iterfind(f".//t:standOff//t:{tag}", NS):
            pid = el.get(XML_ID)
            names = ["".join(n.itertext()).strip() for n in el.findall("t:persName", NS)]
            persons[pid] = dict(name=SHORT.get(pid, names[0] if names else pid),
                                sex=el.get("sex", "mixed" if grp else "unknown"),
                                group=grp)
    return persons


class Resolver:
    def __init__(self, persons):
        self.p, self.unresolved, self.fuzzy = persons, Counter(), {}

    def ids(self, attr) -> list[str]:
        out = []
        for tok in (attr or "").split():
            pid = ID_ALIASES.get(tok.lstrip("#"), tok.lstrip("#$"))
            if not pid:
                continue
            if pid not in self.p:
                m = get_close_matches(pid, list(self.p), n=1, cutoff=0.8)
                if m:
                    self.fuzzy[pid] = m[0]
                    pid = m[0]
                else:
                    self.unresolved[tok] += 1
            out.append(pid)
        return out


def make_labeler(persons):
    def lab(pid):
        if pid in ("себе", "—"):
            return pid
        if pid == "unk":
            return "неизвестно"
        return persons.get(pid, {}).get("name", pid)
    return lab


def gender_of(pid, persons) -> str:
    if pid in ("себе", "—"):
        return pid
    p = persons.get(pid)
    if p is None:
        return "?"
    if p["group"]:
        return "группа"
    return SEX.get(p["sex"], "?")


# ------------------------------------------------------------------ TEI helpers
def text_of(el) -> str:
    return "".join(el.itertext()) if isinstance(el.tag, str) else ""


def chunk(el) -> str:
    return text_of(el) + (el.tail or "")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(DASHES)).strip()


def is_speech(el) -> bool:
    """Speech that bounds narration: <said>, <q who=…>, or a block containing <said>."""
    if not isinstance(el.tag, str):
        return False
    if el.tag == SAID or (el.tag == Q and el.get("who")):
        return True
    return el.find(".//t:said", NS) is not None


def locate(el):
    part = sec = None
    for a in el.iterancestors(DIV):
        if a.get("type") == "section" and sec is None:
            sec = a.get("n")
        if a.get("type") == "part" and part is None:
            part = a.get("n")
    return part, sec


def context_of(el) -> str:
    for a in el.iterancestors(SEG):
        return a.get("type") or "seg"          # dream / imaginary
    return ""


def following(el, who):
    """Narration after el up to the next speech element."""
    seg, names, name_first = el.tail or "", [], False
    t = seg.translate(DASHES)
    lead_empty = "—" in t and t.strip(" ,.!?…—\n\t") == ""
    for i, sib in enumerate(el.itersiblings()):
        if is_speech(sib):
            break
        if isinstance(sib.tag, str) and sib.tag == PERS:
            if i == 0 and lead_empty:
                rs = {ID_ALIASES.get(r.lstrip("#"), r.lstrip("#"))
                      for r in (sib.get("ref") or "").split()}
                name_first = bool(rs & set(who))
            names.append(text_of(sib))
        seg += chunk(sib)
    return seg.translate(DASHES), names, name_first


def narration_before(container, stop_at=None):
    buf, names = container.text or "", []
    for sib in container:
        if sib is stop_at:
            break
        if is_speech(sib):
            buf, names = sib.tail or "", []
            continue
        if isinstance(sib.tag, str) and sib.tag == PERS:
            names.append(text_of(sib))
        buf += chunk(sib)
    return buf.translate(DASHES), names


def preceding(el):
    """Pre-inquit ending in ':' — in the same block or at the end of the previous <p>."""
    parent = el.getparent()
    buf, names = narration_before(parent, el)
    if buf.rstrip().endswith(":"):
        return SENT_SPLIT.split(buf.strip())[-1], names
    if buf.strip(" —\n\t") == "" and parent.tag == P:
        prev = parent.getprevious()
        if prev is not None and prev.tag == P:
            buf2, names2 = narration_before(prev)
            if buf2.rstrip().endswith(":"):
                return SENT_SPLIT.split(buf2.strip())[-1], names2
    return None, []


def manual_fix(part, sec, text):
    for (p_, s_, pref), fix in MANUAL_FIXES.items():
        if p_ == part and s_ == sec and text.startswith(pref):
            return fix
    return {}


# ------------------------------------------------------------------ extraction
def extract(root, res: Resolver, include_q: bool):
    utts, cands = [], []
    tags = (SAID, Q) if include_q else (SAID,)
    for el in root.find(".//t:text", NS).iter(*tags):
        inside_said = next(el.iterancestors(SAID), None) is not None
        if el.tag == SAID and inside_said:           # Gospel dialogue inside Sonya's reading
            continue
        if el.tag == Q and not el.get("who"):
            continue
        part, sec = locate(el)
        text = norm(text_of(el))
        fix = manual_fix(part, sec, text)
        who = res.ids(fix.get("who", el.get("who")))
        to = res.ids(fix.get("toWhom", el.get("toWhom")))
        u = dict(uid=len(utts), part=part, section=sec, unit=f"{part}.{sec}",
                 who=";".join(who), toWhom=";".join(to),
                 speaker=who[0] if who else "unk", n_who=len(who),
                 aloud=el.get("aloud"), direct=el.get("direct"),
                 self_addr=bool(who) and set(to) == set(who),
                 level="said" if el.tag == SAID else ("embedded" if inside_said else "recalled"),
                 context=context_of(el), fixed=bool(fix), text=text[:150])
        found = []
        seg, names, name_first = following(el, who)
        m = POST_RE.match(seg)
        if m:
            body = m.group(1).lstrip()
            if body and (body[0].islower() or name_first):
                found.append(("post", "dash" if body[0].islower() else "namefirst",
                              CUT_RE.split(body, 1)[0], names))
        pre, pnames = preceding(el)
        if pre:
            found.append(("pre", "colon", pre, pnames))
        u["has_inquit"] = bool(found)
        utts.append(u)
        for pos, ftype, txt, nm in found:
            txt = norm(txt)
            cands.append({**u, "position": pos, "form_type": ftype, "inquit": txt,
                          "names": [n for n in nm if n in txt]})
    return utts, cands


def utterance_mode(df) -> np.ndarray:
    return np.select([df.aloud.eq("false"), df.self_addr.astype(bool)],
                     ["про себя", "вслух себе"], "реплика")


def explode_addr(df) -> pd.DataFrame:
    """One row per (utterance, addressee); weight 1/n for multiple addressees."""
    recs = []
    for idx, to, self_ in zip(df.index, df.toWhom, df.self_addr):
        ids = [t for t in (to or "").split(";") if t]
        if self_:
            recs.append((idx, "себе", 1.0))
        elif not ids:
            recs.append((idx, "—", 1.0))
        else:
            recs += [(idx, t, 1 / len(ids)) for t in ids]
    e = pd.DataFrame(recs, columns=["idx", "addr", "w"]).set_index("idx")
    return df.join(e)


# ------------------------------------------------------------------ statistics
def keyness(inq, keep, min_freq):
    """Speaker vs all other speakers, per lemma: Fisher + BH, log2 ratio."""
    ct = pd.crosstab(inq.speaker, inq.lemma)
    N, tot, ns = ct.values.sum(), ct.sum(0), ct.sum(1)
    recs = []
    for s in keep:
        for lem in tot.index[tot >= min_freq]:
            a = int(ct.at[s, lem]); b = int(tot[lem] - a)
            _, p = fisher_exact([[a, ns[s] - a], [b, N - ns[s] - b]])
            recs.append(dict(speaker=s, lemma=lem, n=a, n_speaker=int(ns[s]),
                             pm_speaker=a / ns[s] * 1000, pm_rest=b / (N - ns[s]) * 1000,
                             log2_ratio=np.log2(((a + .5) / (ns[s] + 1)) /
                                                ((b + .5) / (N - ns[s] + 1))), p=p))
    k = pd.DataFrame(recs)
    if len(k):
        k["q_bh"] = multipletests(k.p, method="fdr_bh")[1]
    return k


# ------------------------------------------------------------------ plots
def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, dpi=200); plt.close(fig)


def annot_pct(df):
    return np.where(df.values >= 0.5, df.round().astype(int).astype(str).values, "")


def plot_speaker_verbs(inq, keep, lab, out, top_verbs=25):
    ct = pd.crosstab(inq.speaker, inq.lemma).reindex(keep).fillna(0)
    ct.rename(index=lab).to_csv(out / "speaker_verb_counts.csv")
    share = ct.div(ct.sum(1), axis=0) * 100
    cols = ct.sum().nlargest(top_verbs).index
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 4, 0.45 * len(keep) + 2))
    sns.heatmap(share[cols], annot=annot_pct(share[cols]), fmt="", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "% ремарок персонажа"})
    ax.set_yticklabels([f"{lab(s)} ({int(ct.loc[s].sum())})" for s in keep], rotation=0)
    ax.set(xlabel="", ylabel="", title="Глаголы ремарок по говорящему (@who)")
    save(fig, out, "11_speaker_verbs.png")


def plot_speaker_categories(inq, keep, lab, out):
    d = inq[inq.speaker.isin(keep) & ~inq.category.isin(["—", "не речь"])]
    if d.category.nunique() < 2:
        logging.info("No categories yet — plot 12 skipped (fill verb_categories.csv)")
        return
    tab = pd.crosstab(d.speaker, d.category)
    tab = tab.loc[[s for s in keep if s in tab.index]]
    chi2, p, _, exp = chi2_contingency(tab)
    res = (tab - exp) / np.sqrt(exp)
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.45 * len(tab) + 3))
    share = tab.div(tab.sum(1), axis=0) * 100
    sns.heatmap(share, annot=annot_pct(share), fmt="", cmap="Blues", ax=axes[0])
    sns.heatmap(res, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=axes[1])
    for ax in axes:
        ax.set_yticklabels([lab(s) for s in tab.index], rotation=0); ax.set_ylabel("")
    axes[0].set_title("Доля категории, %")
    axes[1].set_title(f"Стандартизованные остатки (χ²={chi2:.0f}, p={p:.2g})")
    save(fig, out, "12_speaker_categories.png")


def plot_keyness(k, keep, lab, out, top=6, qmax=0.1):
    if k.empty:
        return
    cols = 3; rows = int(np.ceil(len(keep) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.6 * rows), squeeze=False)
    for ax, s in zip(axes.flat, keep):
        d = k[(k.speaker == s) & (k["n"] >= 3)].nlargest(top, "log2_ratio")[::-1]
        ax.barh(d.lemma, d.log2_ratio,
                color=["#c44e52" if q < qmax else "#bbbbbb" for q in d.q_bh])
        ax.axvline(0, color="k", lw=.5)
        n = int(k.loc[k.speaker == s, "n_speaker"].max()) if (k.speaker == s).any() else 0
        ax.set_title(f"{lab(s)} (n={n})", fontsize=10)
    for ax in axes.flat[len(keep):]:
        ax.axis("off")
    fig.suptitle(f"Характерные глаголы (log2 ratio vs остальные); красные — q<{qmax}")
    save(fig, out, "13_speaker_keyness.png")


def plot_addressees(ua, keep, lab, out, top_addr=12):
    m = ua[ua.speaker.isin(keep)].pivot_table(index="speaker", columns="addr", values="w",
                                              aggfunc="sum", fill_value=0)
    m.rename(index=lab, columns=lab).to_csv(out / "addressee_matrix_counts.csv")
    cols = m.sum().nlargest(top_addr).index
    share = (m.div(m.sum(1), axis=0)[cols] * 100).reindex(keep).fillna(0)
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 4, 0.45 * len(keep) + 2))
    sns.heatmap(share, annot=annot_pct(share), fmt="", cmap="OrRd", ax=ax,
                cbar_kws={"label": "% реплик говорящего"})
    ax.set_xticklabels([lab(c) for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([f"{lab(s)} ({int(m.loc[s].sum()) if s in m.index else 0})"
                        for s in keep], rotation=0)
    ax.set(xlabel="адресат (@toWhom)", ylabel="говорящий (@who)",
           title="Кому адресованы реплики (все <said>, не только с ремарками)")
    save(fig, out, "14_addressee_matrix.png")


def plot_network(ua, nodes, lab, persons, out, min_w=5):
    try:
        import networkx as nx
    except ImportError:
        logging.warning("networkx not installed — network plot skipped"); return
    sub = ua[ua.speaker.isin(nodes) & ua.addr.isin(nodes) & (ua.speaker != ua.addr)]
    G = nx.DiGraph()
    for (a, b), v in sub.groupby(["speaker", "addr"]).w.sum().items():
        if v >= min_w:
            G.add_edge(a, b, weight=float(v))
    if not G:
        return
    size = ua.groupby("speaker").uid.nunique()
    pos = nx.spring_layout(G, weight="weight", seed=3, k=1.3)
    wmax = max(d["weight"] for *_, d in G.edges(data=True))
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.5, edge_color="#555", arrowsize=12,
                           connectionstyle="arc3,rad=0.12",
                           width=[.5 + 6 * d["weight"] / wmax for *_, d in G.edges(data=True)])
    nx.draw_networkx_nodes(G, pos, ax=ax, alpha=.85,
                           node_size=[40 * np.sqrt(size.get(n, 1)) for n in G],
                           node_color=[SEXCOL[SEX.get(persons.get(n, {}).get("sex"), "other")]
                                       for n in G])
    nx.draw_networkx_labels(G, pos, {n: lab(n) for n in G}, font_size=9, ax=ax)
    ax.set_title(f"Кто с кем говорит (рёбра ≥ {min_w} реплик; толщина — число реплик)")
    ax.axis("off"); save(fig, out, "15_network.png")


def plot_profiles(utts, keep, lab, out):
    g = utts[utts.speaker.isin(keep)].groupby("speaker")
    prof = pd.DataFrame({
        "реплик": g.size(),
        "с ремаркой автора, %": g["has_inquit"].mean() * 100,
        "про себя (aloud=false), %": g["aloud"].apply(lambda s: (s == "false").mean() * 100),
        "к себе (who=toWhom), %": g["self_addr"].mean() * 100,
    }).reindex(keep)
    prof.rename(index=lab).to_csv(out / "speaker_profiles.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.4 * len(keep) + 2), sharey=True)
    for ax, col in zip(axes, prof.columns):
        ax.barh([lab(s) for s in prof.index][::-1], prof[col][::-1], color="#5b8db8")
        ax.set_title(col)
    save(fig, out, "16_speaker_profiles.png")


def plot_speaker_rarefaction(inq, keep, lab, out, cap=300, reps=50):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in keep:
        pool = inq.loc[inq.speaker == s, "lemma"].to_numpy()
        if len(pool) < 5:
            continue
        sizes = np.unique(np.linspace(5, min(len(pool), cap), 15).astype(int))
        ax.plot(sizes, [np.mean([len(set(rng.choice(pool, n, replace=False)))
                                 for _ in range(reps)]) for n in sizes], label=lab(s))
    ax.set(xlabel="ремарок в выборке", ylabel="разных глаголов",
           title="Лексическое разнообразие ремарок по говорящему (кривые разрежения)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(fig, out, "17_speaker_rarefaction.png")


def plot_focus(ia, focus, lab, out, mode, top_rows=7, top_verbs=15):
    """mode='from': focus speaker's verbs by addressee; 'to': verbs of others addressing focus."""
    if mode == "from":
        d, key = ia[ia.speaker == focus], "addr"
        title = f"{lab(focus)}: глаголы ремарок по адресатам"
    else:
        d, key = ia[(ia.addr == focus) & (ia.speaker != focus)], "speaker"
        title = f"Как вводятся реплики, обращённые к: {lab(focus)}"
    if d.empty:
        logging.info("plot_focus(%s, %s): no data", focus, mode); return
    rows = d.groupby(key).w.sum().nlargest(top_rows).index
    d = d[d[key].isin(rows)]
    m = d.pivot_table(index=key, columns="lemma", values="w", aggfunc="sum", fill_value=0)
    m = m.loc[rows]
    m.rename(index=lab).to_csv(out / f"18_{mode}_{focus}.csv")
    cols = m.sum().nlargest(top_verbs).index
    share = m.div(m.sum(1), axis=0)[cols] * 100
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 4, 0.5 * len(rows) + 2))
    sns.heatmap(share, annot=annot_pct(share), fmt="", cmap="YlOrBr", ax=ax,
                cbar_kws={"label": "% ремарок строки"})
    ax.set_yticklabels([f"{lab(r)} ({m.loc[r].sum():.0f})" for r in rows], rotation=0)
    ax.set(xlabel="", ylabel="адресат" if mode == "from" else "говорящий", title=title)
    save(fig, out, f"18_{mode}_{focus}.png")


def plot_mode_verbs(inq, lab, out, focus=None, top_verbs=20):
    d = inq if focus is None else inq[inq.speaker == focus]
    tab = pd.crosstab(d["mode"], d.lemma)
    tab = tab.reindex([m for m in MODE_ORDER if m in tab.index])
    if len(tab) < 2:
        return
    cols = tab.sum().nlargest(top_verbs).index
    share = tab.div(tab.sum(1), axis=0)[cols] * 100
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 4, 0.6 * len(tab) + 2))
    sns.heatmap(share, annot=annot_pct(share), fmt="", cmap="PuBu", ax=ax,
                cbar_kws={"label": "% ремарок режима"})
    ax.set_yticklabels([f"{m} ({int(tab.loc[m].sum())})" for m in share.index], rotation=0)
    who = "все говорящие" if focus is None else lab(focus)
    ax.set(xlabel="", ylabel="", title=f"Глаголы по режиму речи — {who}")
    save(fig, out, f"19_mode_verbs{'_' + focus if focus else ''}.png")


def plot_gender_dyads(ua, ia, persons, out):
    g = lambda pid: gender_of(pid, persons)
    d = ua.assign(sg=ua.speaker.map(g), ag=ua.addr.map(g))
    d = d[d.sg.isin(["masc", "femn"])]
    m = d.pivot_table(index="sg", columns="ag", values="w", aggfunc="sum", fill_value=0)
    m.to_csv(out / "gender_dyads_counts.csv")
    share = m.div(m.sum(1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(1.1 * share.shape[1] + 3, 3.5))
    sns.heatmap(share, annot=True, fmt=".1f", cmap="Purples", ax=ax)
    ax.set(xlabel="род адресата", ylabel="род говорящего",
           title="Говорящий × адресат по полу, % реплик (все <said>)")
    save(fig, out, "20_gender_dyads.png")

    di = ia.assign(dyad=ia.speaker.map(g) + "→" + ia.addr.map(g))
    di = di[di.dyad.str.match(r"^(masc|femn)→(masc|femn)$")]
    di.pivot_table(index="lemma", columns="dyad", values="w", aggfunc="sum",
                   fill_value=0).sort_values(by=list(di.dyad.unique()), ascending=False) \
      .to_csv(out / "verbs_by_gender_dyad.csv")


# ------------------------------------------------------------------ gender comparison via @who
def run_gender_compare(inq, out, min_freq):
    """Reuse inquit_gender2.compare & plots with gender = speaker's TEI sex."""
    from inquit_gender2 import compare, plot_rest, plot_scatter, plot_diff, plot_manner
    g_out = out / "gender_tei"; g_out.mkdir(exist_ok=True)
    df = inq.assign(chapter=inq.section,
                    layout=np.where(inq["mode"] == "реплика", "реплика", "внутр./себе"))
    t, mf = compare(df, g_out, min_freq)
    plot_rest(df, t, mf, g_out)
    plot_scatter(t, g_out, min_freq)
    plot_diff(t, g_out, min_freq)
    plot_manner(df, g_out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml")
    ap.add_argument("-o", "--outdir", default="results_who")
    ap.add_argument("--spacy-model", default="ru_core_news_lg")
    ap.add_argument("--n-process", type=int, default=1)
    ap.add_argument("--categories", default="/inquit/verb_categories.csv")
    ap.add_argument("--min-freq", type=int, default=5, help="min lemma freq for tests")
    ap.add_argument("--min-speaker", type=int, default=30, help="min inquits per speaker")
    ap.add_argument("--top-speakers", type=int, default=12)
    ap.add_argument("--focus", nargs="*", default=["rrr", "ssm", "porpe", "ais"])
    ap.add_argument("--network-min", type=int, default=5)
    ap.add_argument("--speech-only", action="store_true")
    ap.add_argument("--aloud-only", action="store_true",
                    help="drop aloud=false and self-addressed utterances")
    ap.add_argument("--include-q", action="store_true", help="include <q who=…>")
    ap.add_argument("--no-gender-compare", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    root = etree.parse(args.xml).getroot()
    persons = load_persons(root)
    res, lab = Resolver(persons), make_labeler(persons)
    utts, cands = extract(root, res, args.include_q)
    logging.info("Persons: %d | utterances: %d | inquit candidates: %d",
                 len(persons), len(utts), len(cands))
    if res.unresolved:
        pd.Series(res.unresolved, name="count").to_csv(out / "unresolved_ids.csv")
        logging.warning("Unresolved IDs: %s", dict(res.unresolved))
    if res.fuzzy:
        pd.Series(res.fuzzy, name="matched_to").to_csv(out / "fuzzy_id_matches.csv")
        logging.warning("Fuzzy-matched IDs (check!): %s", res.fuzzy)

    utts = pd.DataFrame(utts)
    utts["mode"] = utterance_mode(utts)

    nlp = spacy.load(args.spacy_model, exclude=["ner"])
    rows = []
    for c, doc in zip(cands, nlp.pipe((c["inquit"] for c in cands),
                                      batch_size=256, n_process=args.n_process)):
        r = analyse(doc, c)
        if r:
            rows.append({**c, **r, "names": "; ".join(c["names"])})
    inq = pd.DataFrame(rows).rename(columns={"gender": "gender_morph"})
    inq["mode"] = utterance_mode(inq)
    inq["speaker_name"] = inq.speaker.map(lab)
    inq["gender"] = [("plur" if n > 1 else gender_of(s, persons))
                     for s, n in zip(inq.speaker, inq.n_who)]
    inq["gender"] = inq.gender.replace({"группа": "plur", "?": "unk"})

    cats = sync_categories(inq, Path(args.categories))
    inq["category"] = inq.lemma.map(cats).fillna("—")

    # parser vs annotation check (before filtering)
    conf = inq[inq.gender_morph.isin(["masc", "femn"]) & inq.gender.isin(["masc", "femn"])
               & (inq.gender_morph != inq.gender)]
    conf.sort_values(["gender_source", "part", "section"])[
        ["part", "section", "speaker_name", "who", "toWhom", "gender", "gender_morph",
         "gender_source", "form", "inquit", "text"]].to_csv(
        out / "conflicts_morph_vs_tei.csv", index=False)
    logging.info("Morph vs TEI gender conflicts: %d of %d", len(conf),
                 inq.gender_morph.isin(["masc", "femn"]).sum())

    if args.speech_only:
        inq = inq[~inq.category.isin(["—", "не речь"])]
    if args.aloud_only:
        inq = inq[inq["mode"] == "реплика"]
        utts = utts[utts["mode"] == "реплика"]
    inq = inq.reset_index(drop=True)
    utts = utts.reset_index(drop=True)
    inq.to_csv(out / "inquits_who.csv", index=False)
    utts.to_csv(out / "utterances.csv", index=False)

    real = lambda s: s != "unk" and not persons.get(s, {}).get("group", True)
    n_inq = inq.speaker.value_counts()
    keep = [s for s in n_inq.index if real(s) and n_inq[s] >= args.min_speaker]
    keep = keep[:args.top_speakers]
    if not keep:
        logging.error("No speaker reaches --min-speaker=%d", args.min_speaker); return
    logging.info("Speakers analysed: %s", [lab(s) for s in keep])

    k = keyness(inq, keep, args.min_freq)
    if len(k):
        k.assign(speaker_name=k.speaker.map(lab)).sort_values(["speaker", "log2_ratio"],
            ascending=[True, False]).to_csv(out / "speaker_keyness.csv", index=False)

    ua, ia = explode_addr(utts), explode_addr(inq)
    n_utt = utts.speaker.value_counts()
    net_nodes = [s for s in n_utt.index if real(s)][:15]

    plot_speaker_verbs(inq, keep, lab, out)
    plot_speaker_categories(inq, keep, lab, out)
    plot_keyness(k, keep, lab, out)
    plot_addressees(ua, keep, lab, out)
    plot_network(ua, net_nodes, lab, persons, out, args.network_min)
    plot_profiles(utts, keep, lab, out)
    plot_speaker_rarefaction(inq, keep, lab, out)
    for f in args.focus:
        plot_focus(ia, f, lab, out, "from")
        plot_focus(ia, f, lab, out, "to")
        plot_mode_verbs(inq, lab, out, focus=f)
    plot_mode_verbs(inq, lab, out)
    plot_gender_dyads(ua, ia, persons, out)
    if not args.no_gender_compare:
        run_gender_compare(inq, out, args.min_freq)
    logging.info("Done: %s", out.resolve())


if __name__ == "__main__":
    main()
