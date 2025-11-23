import math
from typing import Dict, Set

EPS = 1e-9

def compute_scores(
        ngram_freq: Dict[str, int],
        unigram_word_freq: Dict[str, int],
        context_left: Dict[str, Set[str]],
        context_right: Dict[str, Set[str]], 
        min_freq: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
):
    # Validate inputs early to give a clear error message
    if not isinstance(ngram_freq, dict):
        raise TypeError(f"compute_scores: ngram_freq must be dict, got {type(ngram_freq).__name__}")
    if not isinstance(unigram_word_freq, dict):
        raise TypeError(f"compute_scores: unigram_word_freq must be dict, got {type(unigram_word_freq).__name__}")
    if not isinstance(context_left, dict) or not isinstance(context_right, dict):
        raise TypeError("compute_scores: context_left and context_right must be dicts")

    scores: Dict[str, float] = {}

    # Precompute total unigram frequency
    total_unigram = sum(unigram_word_freq.values()) + EPS

    for ngram, f in ngram_freq.items():
        if f < min_freq:
            continue

        # Frequency component
        freq_term = math.log(f + EPS)

        # Compression Gain: Approximate by number of words in ngram
        if " " in ngram:
            len_ngram = len(ngram.split(" "))
        else:
            len_ngram = max(2, len(ngram))  # at least 2 for char ngrams
        
        compression_gain = max(0, len_ngram - 1)

        # PMI-like component
        parts = ngram.split(" ") if " " in ngram else [ngram]
        # Log P(ngram) = log(f) - log(total)
        log_p_ngram = math.log(f + EPS) - math.log(total_unigram)
        # Log P(parts)= sum(log(unigram_count) - log(total))
        log_p_parts = 0.0
        for w in parts:
            uw = unigram_word_freq.get(w, EPS)
            log_p_parts += (math.log(uw + EPS) - math.log(total_unigram))
        pmi = log_p_ngram - log_p_parts

        # Context Diversity
        L = context_left.get(ngram, set())
        R = context_right.get(ngram, set())
        num_contexts = len(L) + len(R) + 1
        context_entropy = math.log(num_contexts + EPS)

        # compute final score as a float (not a set)
        score = (alpha * freq_term) + (beta * compression_gain) + (gamma * pmi) - (delta * context_entropy)
        scores[ngram] = score

    return scores