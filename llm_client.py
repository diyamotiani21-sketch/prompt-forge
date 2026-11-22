"""LLM backends. Set PROVIDER=mock | anthropic | openai (env var).
Keys: ANTHROPIC_API_KEY / OPENAI_API_KEY. Model via MODEL env var."""
import os, random, re, time, json

DATASET = json.load(open(os.path.join(os.path.dirname(__file__), "data", "dataset.json")))


def _mock(prompt):
    """Offline simulator for testing without an API key.
    NOT real model output.
    """

    if "[[JUDGE]]" in prompt:
        ans = prompt.split("ANSWER:")[-1]
        base = 1 + min(2, len(ans.split()) // 60) + (1 if "\n" in ans else 0)
        j = lambda: max(1, min(5, base + random.choice([0, 0, 1])))
        return '{"correctness": %d, "age_fit": %d, "structure": %d, "completeness": %d}' % (j(), j(), j(), j())

    if "[[LADDER]]" in prompt:
        match = re.search(
            r"(?:Tell me about|Explain)\s+(.+?)(?=\s+to\s+|[.!?]|$)",
            prompt,
            re.IGNORECASE
        )

        topic = match.group(1).strip() if match else "the given topic"

        audience_match = re.search(
            r"\bto\s+(.+?)(?:[.!?]|$)",
            prompt,
            re.IGNORECASE
        )

        audience = audience_match.group(1).strip() if audience_match else "the reader"

        lines = [
            f"This explanation is about {topic}.",
            f"Key idea: Understand the basic meaning of {topic}.",
            f"Important details: Explore the main concepts related to {topic}.",
            f"Practical use: Learn how {topic} is relevant in real life.",
            f"Audience: This explanation is intended for {audience}.",
        ]

        n = min(len(prompt.split()) // 4, len(lines))
        return "\n".join(lines[:max(1, n)])

    text = prompt.split("Input:")[-1].split("Output:")[0].strip()

    task = next(
        (t for t, rows in DATASET.items()
         for r in rows if r["input"] == text),
        None
    )

    if task is None and "Solve the word problem." in prompt:
        task = "reasoning_qa"

    if task is None and "Classify the sentiment" in prompt:
        task = "sentiment"

    if task is None and "Summarize the passage" in prompt:
        task = "summarization"

    gold = next(
        (r["answer"] for r in DATASET.get(task, [])
         if r["input"] == text),
        "unknown"
    )

    # Calculate change for differently worded pen questions
    if task == "reasoning_qa" and gold == "unknown":
        numbers = re.findall(r"\d+(?:\.\d+)?", text)

        if "pen" in text.lower() and len(numbers) >= 3:
            price = float(numbers[0])
            quantity = float(numbers[1])
            paid = float(numbers[2])
            gold = str(paid - price * quantity)

            if gold.endswith(".0"):
                gold = gold[:-2]

    p_ok = 0.55

    if "step by step" in prompt:
        p_ok += 0.25

    if "Examples:" in prompt:
        p_ok += 0.15

    if prompt.startswith("You are"):
        p_ok += 0.08

    ok = random.random() < min(p_ok, 0.97)

    if task == "reasoning_qa":
        ans = gold

       
        if ans == "unknown":
            q = text.lower()
            numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]

            if len(numbers) >= 2:
                
                if any(w in q for w in ["change", "remaining", "left", "spends", "spent"]):
                    if len(numbers) >= 3:
                        result = numbers[2] - (numbers[0] * numbers[1])
                    else:
                        result = numbers[0] - numbers[1]

                elif any(w in q for w in ["divided equally", "per person", "share equally", "shared equally", "shared among"]):
                    result = numbers[0] / numbers[1]

                elif any(w in q for w in ["each", "per item", "per notebook", "per pen"]):
                    result = numbers[0] * numbers[1]

                elif any(w in q for w in ["less than", "difference", "subtract"]):
                    result = numbers[0] - numbers[1]

                elif any(w in q for w in ["total", "altogether", "in all", "sum", "combined"]):
                    result = sum(numbers)

                else:
                    result = None

                if result is not None:
                    ans = str(int(result)) if result.is_integer() else str(result)


            if ans != "unknown" and ans.endswith(".0"):
                ans = ans[:-2]

        if "step by step" in prompt:
            return f"Working it out step by step...\nAnswer: {ans}"
        return f"Answer: {ans}"
    if task == "sentiment":
        review = text.lower()

        positive_words = [
            "great", "excellent", "amazing", "good", "happy",
            "love", "helpful", "quality", "recommend", "perfect"
        ]

        negative_words = [
            "terrible", "bad", "poor", "broken", "worst",
            "unhelpful", "disappointed", "waste", "stopped", "awful"
        ]

        positive_count = sum(word in review for word in positive_words)
        negative_count = sum(word in review for word in negative_words)

        return "negative" if negative_count > positive_count else "positive"

    if task == "summarization":
        if "overfitting" in text.lower():
            return (
                "Machine learning models learn patterns from data, "
                "but overfitting makes them memorize training data "
                "and perform poorly on unseen examples."
            )

        return text.split(".")[0].strip() + "."

    return gold if ok else " ".join(gold.split()[:random.randint(3, 6)])

def call_llm(prompt, temperature=0.7):
    """Returns (text, latency_seconds)."""
    provider = os.getenv("PROVIDER", "mock")
    t0 = time.time()
    if provider == "mock":
        out = _mock(prompt)
    elif provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("MODEL", "claude-haiku-4-5"), max_tokens=500, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        out = r.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o-mini"), temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        out = r.choices[0].message.content
    else:
        raise ValueError(f"Unknown PROVIDER {provider}")
    return out, time.time() - t0
