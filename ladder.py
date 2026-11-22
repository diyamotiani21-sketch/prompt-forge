"""Prompt Ladder: one task, five prompt versions, rubric-scored.
Usage: python ladder.py --topic photosynthesis --audience "a Class 8 student" --runs 3"""
import argparse, json, re, pandas as pd
from llm_client import call_llm

VARIANTS = ["1_vague", "2_detailed", "3_role", "4_examples", "5_format"]
RUBRIC = ["correctness", "age_fit", "structure", "completeness"]

def build(variant, topic, audience):
    tag = "[[LADDER]] "  # marker only used by the offline mock
    if variant == "1_vague":
        return f"{tag}Tell me about {topic}."
    if variant == "2_detailed":
        return (f"{tag}Explain {topic} to {audience}. Cover what it is, why it matters, the inputs and outputs, "
                f"and where it happens. Use simple words and avoid jargon.")
    if variant == "3_role":
        return (f"{tag}You are an experienced science teacher who loves making hard ideas easy. "
                f"Explain {topic} to {audience}. Cover what it is, why it matters, the inputs and outputs, and where it happens.")
    if variant == "4_examples":
        return (f"{tag}Explain {topic} to {audience}. Use a short everyday analogy like this example style: "
                f"'A battery stores energy the way a water tank stores water.' Cover what it is, why it matters, "
                f"the inputs and outputs, and where it happens.")
    if variant == "5_format":
        return (f"{tag}You are an experienced science teacher. Explain {topic} to {audience}. "
                f"Format: 1) a one-line definition, 2) three bullet points (inputs, outputs, where), "
                f"3) one everyday analogy, 4) the word equation, 5) one quiz question with its answer. Under 150 words.")
    raise ValueError(variant)

def judge(answer, topic, audience):
    p = (f"[[JUDGE]] You are a strict teacher. Rate the ANSWER to an explanation of {topic} for {audience} "
         f"from 1 (poor) to 5 (excellent) on: correctness, age_fit (suitable for the audience), structure, completeness. "
         f'Reply with JSON only: {{"correctness":n,"age_fit":n,"structure":n,"completeness":n}}\nANSWER: {answer}')
    out, _ = call_llm(p, temperature=0)
    m = re.search(r"\{.*?\}", out, re.S)
    try:
        d = json.loads(m.group(0)); return {k: float(d[k]) for k in RUBRIC}
    except Exception:
        return {k: float("nan") for k in RUBRIC}

def run_ladder(topic="photosynthesis", audience="a Class 8 student", runs=3):
    rows = []
    for v in VARIANTS:
        prompt = build(v, topic, audience)
        for r in range(runs):
            ans, _ = call_llm(prompt)
            rows.append(dict(variant=v, run=r, prompt=prompt.replace("[[LADDER]] ", ""), answer=ans,
                             words=len(ans.split()), **judge(ans, topic, audience)))
    df = pd.DataFrame(rows); df["total"] = df[RUBRIC].sum(axis=1)
    return df

def summary(df):
    return df.groupby("variant")[RUBRIC + ["total", "words"]].mean().round(2).reset_index()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="photosynthesis"); ap.add_argument("--audience", default="a Class 8 student")
    ap.add_argument("--runs", type=int, default=3); a = ap.parse_args()
    df = run_ladder(a.topic, a.audience, a.runs); df.to_csv("results/ladder_raw.csv", index=False)
    s = summary(df); s.to_csv("results/ladder_summary.csv", index=False); print(s.to_string(index=False))
    # best answer per variant, ready to paste on a slide
    with open("results/ladder_slide.md", "w") as f:
        for v in VARIANTS:
            top = df[df.variant == v].sort_values("total", ascending=False).iloc[0]
            f.write(f"## {v}  (score {top.total}/20)\n**Prompt:** {top.prompt}\n\n**Answer:**\n{top.answer}\n\n")
    print("\nSaved results/ladder_*.csv and results/ladder_slide.md")
