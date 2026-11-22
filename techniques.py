"""Prompting techniques. Each builds a prompt string for (task, input)."""

TASK_INSTRUCTIONS = {
    "reasoning_qa": "Solve the word problem.",
    "sentiment": "Classify the sentiment of the review as positive or negative.",
    "summarization": "Summarize the passage in one sentence.",
}
FINAL_FORMAT = {
    "reasoning_qa": "End with a line 'Answer: <number only>'.",
    "sentiment": "Reply with one word only: positive or negative.",
    "summarization": "Reply with the one-sentence summary only.",
}
EXAMPLES = {
    "reasoning_qa": [("A pen costs 5 rupees. How much do 4 pens cost?", "4 x 5 = 20.\nAnswer: 20")],
    "sentiment": [("Great sound quality, very happy!", "positive"), ("Broke on the first use.", "negative")],
    "summarization": [("Solar panels convert sunlight into electricity. They need little maintenance and reduce power bills over time.",
                       "Solar panels turn sunlight into electricity with low maintenance and savings on bills.")],
}

def zero_shot(task, text):
    return f"{TASK_INSTRUCTIONS[task]}\n{FINAL_FORMAT[task]}\n\nInput: {text}"

def few_shot(task, text):
    ex = "\n\n".join(f"Input: {i}\nOutput: {o}" for i, o in EXAMPLES[task])
    return f"{TASK_INSTRUCTIONS[task]}\n{FINAL_FORMAT[task]}\n\nExamples:\n{ex}\n\nInput: {text}\nOutput:"

def chain_of_thought(task, text):
    return (f"{TASK_INSTRUCTIONS[task]}\nThink step by step before giving the final result.\n"
            f"{FINAL_FORMAT[task]}\n\nInput: {text}")

def role_based(task, text):
    roles = {"reasoning_qa": "a careful maths teacher", "sentiment": "an expert product-review analyst",
             "summarization": "a professional technical editor"}
    return f"You are {roles[task]}. {TASK_INSTRUCTIONS[task]}\n{FINAL_FORMAT[task]}\n\nInput: {text}"

TECHNIQUES = {"zero_shot": zero_shot, "few_shot": few_shot,
              "chain_of_thought": chain_of_thought, "role_based": role_based}
