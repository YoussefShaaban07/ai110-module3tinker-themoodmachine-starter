# The Mood Machine

A sentiment classifier for short, social-media-style posts that labels each one **positive**, **negative**, **neutral** or **mixed**. I built a rule-based model and a scikit-learn baseline side by side to compare *where* each one fails on the same data.

**Stack:** Python, scikit-learn · Built for CodePath AI110 (Module 3). Full write-up in the [model card](model_card.md).

## Results

| Model | Accuracy | What it means |
|---|---|---|
| Rule-based (`mood_analyzer.py`) | **0.62** (10/16) | Fails in predictable, explainable ways: sarcasm, unknown words |
| Logistic regression (`ml_experiments.py`) | 1.00 (16/16) | Training accuracy only. It memorized the dataset and would not hold up on new wording |

The comparison was the point: the rule model's mistakes are systematic and can be traced token by token, while the ML model's perfect score is overfitting on 16 posts.

## What I built

- **Rule-based scoring** with word weights (strong words and emojis count double), **negation** that flips the next sentiment word ("not happy" → negative), and a **mixed** label when positive and negative signals cancel out.
- **Emoji handling**: emojis are split into their own tokens and scored (🔥 positive, 💀 negative).
- **Dataset expansion** from 6 to 16 labeled posts, adding slang ("no cap", "lowkey"), emojis, sarcasm and mixed-emotion posts.
- **`explain()`**, which prints exactly which tokens drove each prediction.
- **A model card** covering failure cases, dialect bias in a fixed English word list, and what a real test set would need.

## Run it

```bash
pip install -r requirements.txt
python main.py            # rule-based model plus an interactive demo
python ml_experiments.py  # scikit-learn baseline
```

## Files

```plaintext
├── dataset.py         # labeled posts and sentiment word lists
├── mood_analyzer.py   # rule-based classifier
├── main.py            # runs the rule-based model and the interactive demo
├── ml_experiments.py  # CountVectorizer + LogisticRegression baseline
├── model_card.md      # evaluation, limitations and ethics write-up
└── requirements.txt
```
