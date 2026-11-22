# PromptForge
Compares prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) on reasoning QA,
sentiment classification and summarization, measuring quality, consistency and token cost.

## Setup
pip install -r requirements.txt
## Run (offline test - mock model, NOT real results)
PROVIDER=mock python experiment.py --runs 3 && python plot_results.py
## Run with a real LLM
export PROVIDER=anthropic ANTHROPIC_API_KEY=...   # or PROVIDER=openai OPENAI_API_KEY=...
python experiment.py --runs 5 && python plot_results.py
## Web UI
streamlit run app.py

## Prompt Ladder (one task, five prompts, rubric-scored)
python ladder.py --topic photosynthesis --runs 3
Outputs: results/ladder_summary.csv, results/ladder_slide.md (paste into slides)
