"""
explainer.py — Context-Aware Bias Explanation Engine
=====================================================
Replaces the old keyword-list approach with a multi-signal analyser
that produces natural, varied, analytical sentences.

No heavy model training required — uses lightweight NLP heuristics
combined with sentence-level pattern analysis and template variation.
"""

import re
import random

# ── Signal dictionaries ───────────────────────────────────────────────────────

# Words that carry strong subjective polarity
NEGATIVE_OPINION = {
    "terrible", "awful", "horrible", "dreadful", "atrocious", "abysmal",
    "pathetic", "disgusting", "unbearable", "dull", "boring", "tedious",
    "disappointing", "catastrophic", "lousy", "rotten", "mediocre",
    "overrated", "insufferable", "laughable", "forgettable", "pointless",
}

POSITIVE_OPINION = {
    "amazing", "brilliant", "fantastic", "phenomenal", "extraordinary",
    "masterpiece", "flawless", "breathtaking", "outstanding", "stunning",
    "incredible", "magnificent", "superb", "excellent", "perfect",
    "riveting", "captivating", "exhilarating", "underrated", "genius",
}

# Intensifiers / amplifiers that push language toward subjectivity
INTENSIFIERS = {
    "absolutely", "completely", "totally", "utterly", "entirely",
    "obviously", "clearly", "undoubtedly", "certainly", "definitely",
    "without question", "beyond doubt", "unquestionably", "by far",
    "easily", "simply", "just", "pure", "sheer",
}

# Generalisation signals — broad unsupported claims
GENERALISATION_PATTERNS = [
    r"\beveryone\b", r"\bno one\b", r"\bnothing\b", r"\beverything\b",
    r"\balways\b", r"\bnever\b", r"\ball\b", r"\bnone\b",
    r"\bany\b.{0,20}\bcould\b", r"\bcritics are\b", r"\bpeople always\b",
    r"\bthe audience\b.{0,30}\bwill\b",
]

# Accusatory / blame patterns
ACCUSATORY_PATTERNS = [
    r"\bclearly biased\b", r"\bobviously wrong\b", r"\bdeliberately\b",
    r"\bintentionally\b", r"\bconspiracy\b", r"\bagenda\b",
    r"\bbought\b.{0,20}\breview", r"\bpaid off\b", r"\bcorrupt\b",
    r"\bblind\b.{0,15}\bcritic", r"\bdon.t know\b.{0,20}\btalking about\b",
]

# Subjective first-person markers
SUBJECTIVE_PHRASES = [
    "i think", "i believe", "i feel", "i found", "i thought",
    "in my opinion", "to me", "for me", "personally", "i reckon",
    "i am sure", "i know", "i can tell",
]

# Superlative / extreme claim patterns
SUPERLATIVE_PATTERNS = [
    r"\bworst\b", r"\bbest\b", r"\bmost\b.{0,10}\bever\b",
    r"\bgreatest\b.{0,20}\bof all time\b", r"\bgoat\b",
    r"\bmost\b.{0,8}\boverrated\b", r"\bnothing\b.{0,20}\bcompares\b",
]

# ── Emotion → analytical description map ─────────────────────────────────────

EMOTION_DESCRIPTIONS = {
    "anger":       "frustration and irritation",
    "disgust":     "strong disapproval or revulsion",
    "fear":        "anxiety or apprehension",
    "sadness":     "disappointment or sorrow",
    "joy":         "enthusiasm and excitement",
    "surprise":    "astonishment or disbelief",
    "admiration":  "admiration and favouritism",
    "annoyance":   "irritation and impatience",
    "disapproval": "strong disapproval",
    "excitement":  "heightened enthusiasm",
    "love":        "intense personal affection",
    "optimism":    "over-optimistic framing",
    "pessimism":   "overly pessimistic framing",
    "pride":       "personal pride or possessiveness",
    "remorse":     "regret or guilt",
    "grief":       "deep sorrow",
    "curiosity":   "inquisitive but one-sided framing",
    "relief":      "bias from relief or low expectations",
    "confusion":   "uncertain or conflicted judgement",
    "gratitude":   "excessive gratitude skewing objectivity",
    "amusement":   "dismissive humour undermining objectivity",
    "caring":      "personal emotional investment",
    "desire":      "desire influencing objectivity",
    "embarrassment": "personal embarrassment coloring assessment",
    "nervousness": "anxiety distorting the evaluation",
    "realization": "selective insight presented as fact",
}

# ── Template pools (varied phrasing per signal) ───────────────────────────────

OPINION_TEMPLATES = [
    "The text employs strong {polarity} language — {words} — presenting a subjective stance as though it were an objective fact.",
    "Words such as {words} carry heavy {polarity} connotations, signalling a personal opinion rather than an impartial assessment.",
    "The use of emotionally loaded {polarity} terms like {words} indicates the author's personal bias rather than balanced criticism.",
    "Charged {polarity} descriptors ({words}) reveal a subjective viewpoint presented without factual grounding.",
]

INTENSIFIER_TEMPLATES = [
    "Intensifiers such as {words} amplify the subjective claim beyond what evidence could support, pushing the statement into biased territory.",
    "The presence of amplifying language ({words}) exaggerates the assertion and removes the nuance expected from objective analysis.",
    "Words like {words} strengthen the opinion with unwarranted certainty, a common feature of biased writing.",
]

GENERALISATION_TEMPLATES = [
    "The text makes a broad, unsupported generalisation — statements phrased in absolute terms ('{phrase}') disregard individual variation and suggest opinionated bias.",
    "An absolute claim using '{phrase}' overgeneralises without evidence, which is a hallmark of biased reasoning.",
    "Sweeping assertions such as '{phrase}' present a personal belief as a universal truth, undermining objectivity.",
]

ACCUSATORY_TEMPLATES = [
    "The text contains an accusatory claim ('{phrase}') that attributes negative intent to others without providing supporting evidence.",
    "Phrases like '{phrase}' levy blame without substantiation, reflecting a one-sided perspective.",
    "An accusation captured in '{phrase}' introduces personal judgement that goes beyond a neutral critique.",
]

SUBJECTIVE_TEMPLATES = [
    "First-person framing ('{phrase}') explicitly marks the content as personal opinion, separating it from objective analysis.",
    "The use of '{phrase}' anchors the evaluation in the author's own perspective rather than verifiable fact.",
    "Self-referential language such as '{phrase}' signals that the assessment is subjective and not independently verifiable.",
]

SUPERLATIVE_TEMPLATES = [
    "Superlative or extreme claims ('{phrase}') leave no room for nuance, a strong indicator of biased framing.",
    "The use of absolute superlatives like '{phrase}' exaggerates the assessment and is characteristic of opinion-driven writing.",
    "Extreme language such as '{phrase}' polarises the evaluation, which is inconsistent with balanced, evidence-based criticism.",
]

EMOTION_TEMPLATES = [
    "The emotional undertone of {description} detected in this text colours the interpretation and moves it away from an impartial viewpoint.",
    "An underlying sentiment of {description} influences the tone, suggesting the author's personal feelings are shaping the evaluation.",
    "The presence of {description} in the writing indicates that emotional reactions — rather than objective criteria — are driving the assessment.",
]

CONFIDENCE_TEMPLATES = [
    "The model classified this text as biased with {confidence}% confidence, reflecting the density of subjective signals identified above.",
    "Overall, the combination of signals above led the classifier to flag this text as biased (confidence: {confidence}%).",
]

NEUTRAL_TEMPLATES = [
    "The language used is largely descriptive and factual, with minimal subjective or emotionally charged expressions detected.",
    "No strong indicators of personal opinion, generalisation, or emotional exaggeration were found — the text reads as relatively objective.",
    "The text presents information without notable opinion-driven language, superlatives, or unsupported claims.",
]


# ── Core analysis functions ───────────────────────────────────────────────────

def _find_opinion_words(text_lower):
    neg = [w for w in NEGATIVE_OPINION if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
    pos = [w for w in POSITIVE_OPINION if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
    return neg, pos


def _find_intensifiers(text_lower):
    return [w for w in INTENSIFIERS if w in text_lower]


def _find_generalisations(text_lower):
    matches = []
    for pattern in GENERALISATION_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            matches.append(m.group(0).strip())
    return matches


def _find_accusatory(text_lower):
    matches = []
    for pattern in ACCUSATORY_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            matches.append(m.group(0).strip())
    return matches


def _find_subjective(text_lower):
    return [p for p in SUBJECTIVE_PHRASES if p in text_lower]


def _find_superlatives(text_lower):
    matches = []
    for pattern in SUPERLATIVE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            matches.append(m.group(0).strip())
    return matches


def _format_word_list(words):
    """Return a quoted comma-separated string of words."""
    return ", ".join(f'"{w}"' for w in words[:4])   # cap at 4 for readability


# ── Public API ────────────────────────────────────────────────────────────────

def generate_explanation(text: str, emotion: str, prediction: str, confidence: float) -> list:
    """
    Analyse *text* across multiple signal dimensions and return a list of
    2–4 human-readable analytical sentences explaining the bias verdict.

    Parameters
    ----------
    text        : raw input string
    emotion     : emotion label from GoEmotions API (or "Unknown")
    prediction  : "Biased" or "Neutral" from the ML model
    confidence  : model confidence as a percentage float (0–100)

    Returns
    -------
    list of str — 2–4 explanation sentences
    """
    text_lower = text.lower()
    explanations = []

    # ── Short-circuit for neutral predictions ────────────────────────────────
    if prediction == "Neutral":
        explanations.append(random.choice(NEUTRAL_TEMPLATES))
        if confidence >= 80:
            explanations.append(
                f"The classifier assigned a high confidence of {confidence}% to this "
                "neutral verdict, suggesting the text is largely free of subjective markers."
            )
        return explanations

    # ── Signal 1: Opinion words ───────────────────────────────────────────────
    neg_words, pos_words = _find_opinion_words(text_lower)

    if neg_words:
        tmpl = random.choice(OPINION_TEMPLATES)
        explanations.append(tmpl.format(
            polarity="negative", words=_format_word_list(neg_words)
        ))
    elif pos_words:
        tmpl = random.choice(OPINION_TEMPLATES)
        explanations.append(tmpl.format(
            polarity="positive", words=_format_word_list(pos_words)
        ))

    # ── Signal 2: Intensifiers ────────────────────────────────────────────────
    intensifiers = _find_intensifiers(text_lower)
    if intensifiers and len(explanations) < 3:
        tmpl = random.choice(INTENSIFIER_TEMPLATES)
        explanations.append(tmpl.format(words=_format_word_list(intensifiers)))

    # ── Signal 3: Generalisations ─────────────────────────────────────────────
    generalisations = _find_generalisations(text_lower)
    if generalisations and len(explanations) < 3:
        tmpl = random.choice(GENERALISATION_TEMPLATES)
        explanations.append(tmpl.format(phrase=generalisations[0]))

    # ── Signal 4: Accusatory language ────────────────────────────────────────
    accusatory = _find_accusatory(text_lower)
    if accusatory and len(explanations) < 3:
        tmpl = random.choice(ACCUSATORY_TEMPLATES)
        explanations.append(tmpl.format(phrase=accusatory[0]))

    # ── Signal 5: First-person subjective markers ─────────────────────────────
    subjective = _find_subjective(text_lower)
    if subjective and len(explanations) < 3:
        tmpl = random.choice(SUBJECTIVE_TEMPLATES)
        explanations.append(tmpl.format(phrase=subjective[0]))

    # ── Signal 6: Superlatives / extreme claims ───────────────────────────────
    superlatives = _find_superlatives(text_lower)
    if superlatives and len(explanations) < 3:
        tmpl = random.choice(SUPERLATIVE_TEMPLATES)
        explanations.append(tmpl.format(phrase=superlatives[0]))

    # ── Signal 7: Emotion (from GoEmotions API) ───────────────────────────────
    if emotion and emotion.lower() not in ("unknown", "neutral") and len(explanations) < 4:
        description = EMOTION_DESCRIPTIONS.get(emotion.lower(), f"{emotion.lower()} emotion")
        tmpl = random.choice(EMOTION_TEMPLATES)
        explanations.append(tmpl.format(description=description))

    # ── Fallback if nothing fired ─────────────────────────────────────────────
    if not explanations:
        explanations.append(
            "The model detected subtle patterns — such as implicit framing or tone — "
            "that collectively indicate a non-neutral perspective, even without explicit opinion words."
        )

    # ── Confidence closing sentence (append if we have room) ─────────────────
    if len(explanations) <= 3 and confidence >= 75:
        tmpl = random.choice(CONFIDENCE_TEMPLATES)
        explanations.append(tmpl.format(confidence=confidence))

    return explanations[:4]   # hard cap at 4 points