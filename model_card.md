# Model Card: Mood Machine

This model card covers **both** versions of the Mood Machine classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

I built and tested both so I could compare where handwritten rules and a learned
model fail on the *same* data.

## 1. Model Overview

**Model type:**
I compared both models. The rule based model is the main system I engineered; the
ML model is a small logistic-regression baseline used for comparison.

**Intended purpose:**
Classify short social-media-style posts as one of four moods: `positive`,
`negative`, `neutral`, or `mixed`.

**How it works (brief):**
- *Rule based:* text is lowercased and tokenized (emojis are split into their own
  tokens, surrounding punctuation is stripped). Each sentiment word adds or
  subtracts points; strong words (`love`, `hate`, `terrible`, emojis) count double;
  a negation word (`not`, `never`, `no`…) flips the sign of the next sentiment word.
  The final score maps to a label, with a `mixed` label when a post carries both
  positive and negative signal and neither side wins.
- *ML:* posts are turned into bag-of-words counts with `CountVectorizer`, and a
  `LogisticRegression` model learns which words predict which label.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` started with 6 posts and I expanded it to **16** by adding 10 new
posts in `dataset.py` (see `NEW_POSTS` / `NEW_LABELS`). Every added post has one
matching label; an `assert` at the bottom of `dataset.py` keeps the two lists the
same length.

**Labeling process:**
I labeled by reading each post and asking "what does the writer actually feel?"
rather than "what words appear." That distinction matters for sarcasm and emojis.
Posts I found genuinely hard to label:
- `"I absolutely love getting stuck in traffic"` — labeled **negative** because it's
  sarcastic, even though it literally contains "love." A reader who misses the tone
  could reasonably call it positive.
- `"so happy i could cry 😭"` — labeled **positive**, but the 😭 emoji reads negative
  in isolation.
- `"ugh mondays are the worst but coffee helps"` — labeled **mixed**; someone else
  might call it negative.

**Important characteristics of the dataset:**
- Contains slang (`lowkey`, `highkey`, `no cap`, `fire`)
- Contains emojis (🥲 🔥 💀 😭)
- Contains sarcasm (2 posts)
- Contains genuinely mixed feelings (3 posts)
- Posts are short (5–10 words), like real messages

**Possible issues with the dataset:**
It is tiny (16 posts) and slightly imbalanced (more negative/positive than mixed).
It also can't teach a model much about *long* text.

## 3. How the Rule Based Model Works

**Scoring rules:**
- Positive word → `+weight`, negative word → `−weight`.
- `strong_words` (`love`, `hate`, `amazing`, `terrible`, `awful`, `worst`,
  `awesome`) and emojis are weighted **2**; everything else is **1**.
- **Negation:** a negation word flips the sign of the following sentiment word, so
  `"not happy"` scores −1 and `"not bad"` would score +1.
- **Emoji handling:** emojis are split out in `preprocess` and scored like strong
  words (🔥 😊 positive; 💀 😭 :( negative).
- **Thresholds:** `mixed` when both positive and negative signal exist and
  `|score| ≤ 1`; otherwise `>0 positive`, `<0 negative`, `==0 neutral`.

**Strengths:** predictable and fully explainable — `explain()` prints exactly which
tokens drove the score. It handles clear cases, simple negation, and emoji-carrying
posts well.

**Weaknesses:** no understanding of sarcasm, context, or unknown words (see §5).

## 4. How the ML Model Works

**Features used:** bag of words via `CountVectorizer` (raw word counts).

**Training data:** trained on the same `SAMPLE_POSTS` / `TRUE_LABELS`.

**Training behavior:** on the current 16 posts it reaches **1.00 accuracy** — but
this is *training* accuracy, so it is memorizing the words it has seen. When I added
new posts, previously-"solved" examples could flip, because a new post changes which
words the model associates with each label.

**Strengths and weaknesses:** it learns word→label associations automatically (it
"got" the two sarcasm posts that the rule model missed, simply because the exact
words appeared with a negative label in training). But that same behavior is
overfitting: it would almost certainly fail on new wording it never saw, and it
can't explain *why* it chose a label.

## 5. Evaluation

**How I evaluated:** ran `python main.py` (rule based) and `python ml_experiments.py`
(ML) on the labeled posts in `dataset.py`.

- **Rule based accuracy: 0.62** (10 / 16 correct)
- **ML training accuracy: 1.00** (16 / 16 — overfit, see §4)

**Examples of correct rule-based predictions:**
- `"this app is straight fire no cap 🔥"` → positive. Correct: `fire` (+1) and 🔥
  (+2) both fire, score = 3.
- `"I am not happy about this"` → negative. Correct: negation flips `happy` to −1.
- `"Lowkey stressed but highkey proud of myself 🥲"` → mixed. Correct: `proud` (+)
  and `stressed` (−) cancel to 0 with signal on both sides.

**Examples of incorrect rule-based predictions:**
- `"I absolutely love getting stuck in traffic"` → predicted **positive**, true
  **negative**. Sarcasm: the single word `love` dominated; the model has no way to
  see that "getting stuck in traffic" is bad.
- `"just chilling, nothing much going on"` → predicted **positive**, true
  **neutral**. `chilling` is in the positive list, but here it just means "relaxing/
  idle," not an emotional positive.
- `"10/10 would NOT recommend"` → predicted **neutral**, true **negative**.
  `recommend` isn't in the vocabulary, so the negation had nothing to flip.

**How the two models' failures differ:** the ML model got all three of the above
*right* on this dataset — but only because it memorized those exact words. The rule
model's failures are systematic and explainable; the ML model's success is fragile
and would not survive new phrasing.

## 6. Limitations

- The dataset is very small (16 posts) — both accuracy numbers are optimistic.
- The rule model can't detect sarcasm and is blind to any word not in its lists.
- The ML model's 1.00 is training accuracy (overfitting), not real-world accuracy.
- Neither model handles longer text, context, or negation that spans several words.

## 7. Ethical Considerations

- **Distress misclassification:** a post like `"I'm fine 🙂"` could hide real
  distress; a mood tool that labels it "positive" could cause someone to miss a
  cry for help.
- **Overt bias / language style:** the vocabulary is standard English plus a little
  internet slang. Posts written in regional dialects, AAVE, or non-English-influenced
  phrasing would score near 0 (words not in the lists) and get pushed toward
  `neutral`. That is a systematic failure tied to *who wrote the text*, not what it
  says — the model would underserve writers whose style doesn't match my word lists.
- **Covert compliance:** when I asked the AI coding assistant to review my word list,
  it mostly confirmed my choices and suggested *more* words to add — it did **not**
  flag that a fixed English word list systematically disadvantages non-standard
  dialects until I asked directly about bias. It should have raised that on its own.
  I treated its suggestions as hypotheses and verified each against the actual posts.
- **Privacy:** running mood detection on personal messages is sensitive and should
  be opt-in.

## 8. Ideas for Improvement

- Add more labeled data, and hold out a real **test set** instead of reporting
  training accuracy.
- Use TF-IDF instead of raw counts, and add proper emoji/slang normalization.
- Handle multi-word negation and sarcasm cues (e.g., a mismatch between positive
  words and clearly-negative situations).
- Expand vocabulary with dialect-aware terms, or move to a small learned embedding
  model so unseen words still get a signal.
