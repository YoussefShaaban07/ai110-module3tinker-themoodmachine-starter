"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # added while expanding the dataset (slang + emoji signals)
    "proud",
    "hopeful",
    "chilling",
    "fire",      # slang: "that's fire" = good
    ":)",
    "🙂",
    "🔥",
    "😊",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # added while expanding the dataset (slang + emoji signals)
    "worst",
    "broke",
    "ugh",
    ":(",
    "💀",
    "😭",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# ---------------------------------------------------------------------
# Expanded dataset (added by me)
# ---------------------------------------------------------------------
# A mix of slang, emojis, sarcasm, and genuinely-mixed feelings. Several of
# these are deliberately hard to label — I noted the tricky ones so they are
# easy to inspect later against both the rule based and ML models.
NEW_POSTS = [
    "Lowkey stressed but highkey proud of myself 🥲",   # mixed: stress + pride
    "this app is straight fire no cap 🔥",               # positive: slang
    "I absolutely love getting stuck in traffic",        # SARCASM -> really negative
    "meh, it's whatever",                                # neutral / flat
    "wow what a fantastic day, everything broke 💀",     # SARCASM -> really negative
    "just chilling, nothing much going on",              # neutral
    "I'm not sad, just tired",                           # negation + tired -> negative
    "so happy i could cry 😭",                           # positive (crying-happy)
    "ugh mondays are the worst but coffee helps",        # mixed
    "10/10 would NOT recommend",                         # negative (negation)
]

NEW_LABELS = [
    "mixed",
    "positive",
    "negative",   # a friend might argue "positive" if they miss the sarcasm
    "neutral",
    "negative",
    "neutral",
    "negative",
    "positive",   # "😭" reads negative alone, but the sentence is happy
    "mixed",
    "negative",
]

SAMPLE_POSTS.extend(NEW_POSTS)
TRUE_LABELS.extend(NEW_LABELS)

# Safety check: the two lists must always stay aligned.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS ({len(SAMPLE_POSTS)}) and TRUE_LABELS "
    f"({len(TRUE_LABELS)}) must be the same length."
)

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
