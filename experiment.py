"""Run every technique on every task, several times, and save results.
Usage: PROVIDER=mock python experiment.py --runs 3"""
import argparse, json, pandas as pd
from techniques import TECHNIQUES
from llm_client import call_llm, DATASET
from metrics import score, extract_answer, approx_tokens, consistency

def run(runs=3):
    rows = []
    for task, items in DATASET.items():
        for tech, build in TECHNIQUES.items():
            for idx, item in enumerate(items):
                prompt = build(task, item["input"])
                preds, scores, lats, outs = [], [], [], []
                for _ in range(runs):
                    out, lat = call_llm(prompt)
                    preds.append(extract_answer(task, out)); scores.append(score(task, out, item["answer"]))
                    lats.append(lat); outs.append(out)
                rows.append(dict(task=task, technique=tech, item=idx, score=sum(scores)/runs,
                    consistency=consistency(preds), prompt_tokens=approx_tokens(prompt),
                    output_tokens=sum(approx_tokens(o) for o in outs)/runs, latency=sum(lats)/runs))
    return pd.DataFrame(rows)

def summarize(df):
    s = df.groupby(["task", "technique"]).agg(score=("score","mean"), consistency=("consistency","mean"),
        prompt_tokens=("prompt_tokens","mean"), output_tokens=("output_tokens","mean")).round(3).reset_index()
    s["cost_tokens"] = s.prompt_tokens + s.output_tokens
    return s

def best_per_task(s):
    return s.sort_values(["task","score","consistency"], ascending=[True,False,False]).groupby("task").head(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=3); a = ap.parse_args()
    df = run(a.runs); df.to_csv("results/raw.csv", index=False)
    s = summarize(df); s.to_csv("results/summary.csv", index=False)
    print(s.to_string(index=False)); print("\nBest technique per task:\n", best_per_task(s)[["task","technique","score"]].to_string(index=False))
