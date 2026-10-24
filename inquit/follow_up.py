# followup.py — run after speech_who.py; adjust paths if needed
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

R = "inquit/results_who"
df = pd.read_csv(f"{R}/inquits_who.csv")
mf = df[df.gender.isin(["masc", "femn"])]
vt = pd.read_csv(f"{R}/gender_tei/verb_table.csv", index_col=0)

# 1. the six significant verbs: effect, chapter-bootstrap CI, and who produces them
sig = vt[vt.q_bh < .05]
print(sig[["masc", "femn", "pm_masc", "pm_femn", "diff_pm", "diff_lo", "diff_hi", "q_bh"]].round(3))
for lem in sig.index:
    d = mf[mf.lemma == lem]
    top = d.groupby(["gender", "speaker_name"]).size().sort_values(ascending=False)
    print(f"\n{lem}: top speaker share = {top.iloc[0] / len(d):.0%}\n{top.head(5).to_string()}")

# 2. aspect: leave-one-speaker-out and by speech mode
def perf(d):
    t = pd.crosstab(d.gender, d.aspect.eq("perf")).reindex(
        index=["masc", "femn"], columns=[True, False], fill_value=0)
    return (t[True] / t.sum(1)).round(3).to_dict(), fisher_exact(t.values)[1]

print("\n--- leave-one-speaker-out ---")
for s in [None, *mf.speaker_name.value_counts().head(8).index]:
    share, p = perf(mf if s is None else mf[mf.speaker_name != s])
    print(f"{'все' if s is None else 'без ' + s:25} {share}  p={p:.4f}")
print("\n--- by mode ---")
for m, d in mf.groupby("mode"):
    if d.gender.nunique() == 2:
        print(m, len(d), *perf(d))

# 3. speaker as the unit of analysis
sp = (mf.groupby(["speaker_name", "gender"])
        .agg(n=("lemma", "size"),
             perf=("aspect", lambda a: (a == "perf").mean()),
             neutral=("category", lambda c: (c == "нейтральные").mean()))
        .reset_index().query("n >= 20").sort_values("perf"))
print("\n", sp.round(3).to_string(index=False))
for col in ("perf", "neutral"):
    m, f = sp[sp.gender == "masc"][col], sp[sp.gender == "femn"][col]
    if len(f) >= 3:
        print(col, mannwhitneyu(m, f))
