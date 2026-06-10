import re
import sys
sys.path.append('.')
from rag.domain_dict import _CASTE_HINTS
from rapidfuzz import process as _rfprocess, fuzz as _rffuzz

query = "show data of all fakirs and muslims in Jaipur"
stopwords = {"give", "me", "data", "of", "all", "people", "from", "caste", "if", "show", "count", "list", "how", "many", "the", "in", "for", "with", "who", "are", "is", "what", "where", "and", "or", "than", "more", "less", "under", "over", "don't", "dont", "person"}
words = re.sub(r"[^\w\-\s]", "", query).lower().split()
ngrams = []
for n in range(1, 4):
    for i in range(len(words) - n + 1):
        ngram_words = words[i:i+n]
        ngram = " ".join(ngram_words)
        if len(ngram) > 3 and not all(w in stopwords for w in ngram_words) and not any(c.isdigit() for c in ngram):
            ngrams.append(ngram)

seen = set()
mapped = set()
for ngram in ngrams:
    result = _rfprocess.extractOne(
        ngram,
        _CASTE_HINTS.keys(),
        scorer=_rffuzz.ratio,
        score_cutoff=85
    )
    if result:
        match_key, score, _ = result
        hint = _CASTE_HINTS[match_key]
        mapped.update(ngram.split())
        print(f"Match: {ngram!r} -> {match_key!r} (score {score})")
        if hint not in seen:
            seen.add(hint)
            print(f"  Added hint: {hint[:20]}")
        else:
            print(f"  Hint already seen.")

print("Mapped words:", mapped)
