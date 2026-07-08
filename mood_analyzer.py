# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

        # Words that flip the sentiment of the NEXT sentiment word.
        # ("not happy" -> negative, "not bad" -> positive)
        self.negations = {"not", "no", "never", "aint", "ain't", "dont", "don't"}

        # A few words carry more emotional weight than the average signal.
        self.strong_words = {
            "love": 2, "hate": 2, "amazing": 2, "terrible": 2,
            "awful": 2, "worst": 2, "awesome": 2,
        }

        # Emojis / emoticons are treated as strong, standalone signals.
        self.emoji_tokens = {":)", ":(", "🙂", "🔥", "😊", "💀", "😭", "🥲"}

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        # Pull emojis / emoticons out as their own tokens so punctuation
        # stripping does not destroy them (":)" would otherwise become "").
        for emoji in self.emoji_tokens:
            if emoji in cleaned:
                cleaned = cleaned.replace(emoji, f" {emoji} ")

        tokens: List[str] = []
        for raw in cleaned.split():
            if raw in self.emoji_tokens:
                tokens.append(raw)
                continue
            # Strip surrounding punctuation but keep the word (traffic! -> traffic).
            word = raw.strip(".,!?;:\"'()[]")
            if word:
                tokens.append(word)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        return self._analyze(text)["score"]

    def _analyze(self, text: str) -> Dict:
        """
        Core scoring pass. Returns a dict with the score plus the positive and
        negative contributions, so predict_label and explain can reuse it.

        Enhancements implemented here:
          - Word weighting: "love"/"hate"/etc. count double (see strong_words).
          - Emoji signals: 🔥/😊 add, 💀/😭/:( subtract.
          - Negation: a negation word flips the NEXT sentiment word's sign,
            so "not happy" scores negative and "not bad" scores positive.
        """
        tokens = self.preprocess(text)

        score = 0
        pos_total = 0
        neg_total = 0
        negate = False  # set True right after a negation word

        for token in tokens:
            if token in self.negations:
                negate = True
                continue

            weight = self.strong_words.get(token, 2 if token in self.emoji_tokens else 1)

            value = 0
            if token in self.positive_words:
                value = weight
            elif token in self.negative_words:
                value = -weight

            if value != 0:
                if negate:
                    value = -value  # flip the sentiment
                if value > 0:
                    pos_total += value
                else:
                    neg_total += -value
                score += value

            # Negation only affects the immediately following sentiment word.
            negate = False

        return {"score": score, "pos": pos_total, "neg": neg_total}

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        result = self._analyze(text)
        score, pos, neg = result["score"], result["pos"], result["neg"]

        # "mixed" = the text carries BOTH strong positive and negative signal
        # and neither side clearly wins.
        if pos > 0 and neg > 0 and abs(score) <= 1:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate = False

        for token in tokens:
            if token in self.negations:
                negate = True
                continue

            is_pos = token in self.positive_words
            is_neg = token in self.negative_words
            if is_pos or is_neg:
                # After a negation the effective sign is flipped.
                effective_pos = is_pos ^ negate
                label_token = f"NOT {token}" if negate else token
                (positive_hits if effective_pos else negative_hits).append(label_token)
            negate = False

        result = self._analyze(text)
        return (
            f"Score = {result['score']} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
