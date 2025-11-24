import re
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Iterable

WORD_RE = re.compile(r"\w+(?:['’]\w+)?[^\w\s]*|[^\w\s]", re.UNICODE)
# Matches words or single non-whitespace characters

def tokenize_words(line: str) -> List[str]:
    return WORD_RE.findall(line)

def extract_ngram_stats(
    text_lines: Iterable[str],
    max_char_n: int = 6,
    max_word_n: int = 5,
    min_char_len: int = 2,
    min_word_n: int = 2,
):
    char_ngram_freq: Dict[str, int] = defaultdict(int)
    word_ngram_freq: Dict[str, int] = defaultdict(int)
    unigram_word_freq: Dict[str, int] = defaultdict(int)

    context_left: Dict[str, Set[str]] = defaultdict(set)
    context_right: Dict[str, Set[str]] = defaultdict(set)

    for line in text_lines:
        line = line.strip()
        if not line:
            continue

        # Character n-grams
        chars = list(line)
        words = tokenize_words(line)

        for w in words:
            unigram_word_freq[w] += 1
        
        Lc = len(chars)
        for i in range(Lc):
            for n in range(min_char_len, max_char_n + 1):
                if i + n > Lc:
                    break
                ngram = "".join(chars[i:i+n])
                char_ngram_freq[ngram] += 1

                left = chars[i-1] if i > 0 else "<BOS>"
                right = chars[i+n] if i + n < Lc else "<EOS>"
                context_left[ngram].add(left)
                context_right[ngram].add(right)

        Lw = len(words)
        for i in range(Lw):
            for n in range(min_word_n, max_word_n + 1):
                if i + n > Lw:
                    break
                ngram_words = words[i:i+n]
                ngram = " ".join(ngram_words)
                word_ngram_freq[ngram] += 1

                left = words[i-1] if i > 0 else "<BOS>"
                right = words[i+n] if i + n < Lw else "<EOS>"
                context_left[ngram].add(left)
                context_right[ngram].add(right)
    return (
        char_ngram_freq,
        word_ngram_freq,
        unigram_word_freq,
        context_left,
        context_right,
    )