from rapidfuzz.distance import Levenshtein

from .common import normalize


def counts(reference, prediction, casefold=True):
    ref, hyp = normalize(reference, casefold), normalize(prediction, casefold)
    return {
        "char_errors": Levenshtein.distance(ref, hyp),
        "chars": len(ref),
        "word_errors": Levenshtein.distance(ref.split(), hyp.split()),
        "words": len(ref.split()),
        "exact": int(ref == hyp),
        "blank_false_positive_chars": len(hyp) if not ref else 0,
    }


def aggregate(items):
    if not items:
        raise ValueError("Cannot score an empty evaluation")
    sums = {k: sum(x[k] for x in items) for k in items[0]}
    return {
        "n": len(items), **sums,
        "cer": sums["char_errors"] / sums["chars"] if sums["chars"] else None,
        "wer": sums["word_errors"] / sums["words"] if sums["words"] else None,
        "exact_match": sums["exact"] / len(items),
    }
