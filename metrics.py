import re
from collections import Counter

def extract_answer(task, output):
    if task == "reasoning_qa":
        m = re.findall(r"Answer:\s*([-\d.]+)", output)
        if m: return m[-1].rstrip(".")
        nums = re.findall(r"-?\d+\.?\d*", output)
        return nums[-1] if nums else ""
    if task == "sentiment":
        o = output.lower()
        return "negative" if "negative" in o else "positive" if "positive" in o else ""
    return output.strip()

def _tokens(s): return re.findall(r"\w+", s.lower())

def f1_overlap(pred, gold):
    """Word-overlap F1 (simple ROUGE-1-like) for summaries."""
    p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
    common = sum((p & g).values())
    if not common: return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)

def score(task, output, gold):
    """Returns 0..1 quality score."""
    pred = extract_answer(task, output)
    if task == "summarization": return f1_overlap(pred, gold)
    try: return float(abs(float(pred) - float(gold)) < 1e-6) if task == "reasoning_qa" else float(pred == gold)
    except ValueError: return 0.0

def approx_tokens(text): return max(1, int(len(text.split()) * 1.3))

def consistency(preds):
    """Share of runs agreeing with the most common answer (1.0 = fully stable)."""
    return Counter(preds).most_common(1)[0][1] / len(preds)
