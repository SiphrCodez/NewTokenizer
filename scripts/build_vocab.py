import json
from typing import Iterable, List

from tokenizer.base_alphabet import build_base_alphabet
from tokenizer.ngram_extraction import extract_ngram_stats
from tokenizer.scoring import compute_scores
from tokenizer.vocab_builder import (
    build_initial_vocab_from_ngrams,
    train_unigram_vocab,
    export_vocab,
)
from tokenizer.encode_decode import text_to_base_ids

TRAIN_PATH = "data/osb_raw.txt"
VOCAB_OUT_PATH = "artifacts/vocab_v1.jsonl"

# Hyperparams (TUNE THESE)
MAX_CHAR_N = 6
MAX_WORD_N = 5
MIN_CHAR_LEN = 2
MIN_WORD_N = 2

MIN_WORD_FREQ = 10 # Minimum Freq for Word Ngram Candidates
TOP_K_PHRASES = 300_000 # Number of Top Phrase Ngrams to Seed Vocabulary
TOP_K_WORD_UNIGRAM = 50_000 # Number of Words to Seed Vocabulary
TARGET_VOCAB_SIZE = 100_000 # Final Vicab Size Target (Including BASE & Special)
PRUNE_RATE = 0.1 # Fraction of Vocab Pruned Per Iteration
MAX_ITERS = 10 # Max unigram iterations.

def read_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def main():
    # 1. Build Base Alphabet
    print("[BUILD_VOCAB]: Building base alphabet...")
    char2id, id2char = build_base_alphabet()
    print(f"[BUILD_VOCAB]: Base alphabet size: {len(char2id)}")

    # 2. Extract Ngram Stats
    print("[BUILD_VOCAB]: Extracting ngram stats...")
    lines_for_ngrams = read_lines(TRAIN_PATH)
    (
        char_ngram_freq,
        word_ngram_freq,
        unigram_word_freq,
        context_left,
        context_right,
    ) = extract_ngram_stats(
        lines_for_ngrams,
        max_char_n=MAX_CHAR_N,
        max_word_n=MAX_WORD_N,
        min_char_len=MIN_CHAR_LEN,
        min_word_n=MIN_WORD_N,
    )

    print(f"[BUILD_VOCAB]: Char_ngram_freq: {len(char_ngram_freq)}")
    print(f"[BUILD_VOCAB]: Word_ngram_freq: {len(word_ngram_freq)}")
    print(f"[BUILD_VOCAB]: Unigram_word_freq: {len(unigram_word_freq)}")

    # 3. Score Word Ngrams (For Phrase/Word Tokens)
    print("[BUILD_VOCAB]: Scoring word ngrams...")
    scored_word_ngrams = compute_scores(
        ngram_freq=word_ngram_freq,
        unigram_word_freq=unigram_word_freq,
        context_left=context_left,
        context_right=context_right,
        min_freq=MIN_WORD_FREQ,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        delta=0.5,
    )
    print(f"[BUILD_VOCAB]: Scored_word_ngrams: {len(scored_word_ngrams)}")

    # 4. Building Initial Vocab: Special + Base + Top-K Word Ngrams
    print("[BUILD_VOCAB]: Building initial vocab from ngrams...")
    vocab = build_initial_vocab_from_ngrams(
        char2id=char2id,
        id2char=id2char,
        scored_word_ngrams=scored_word_ngrams,
        unigram_word_freq=unigram_word_freq,
        char_ngram_freq=char_ngram_freq,
        top_k_phrases=TOP_K_PHRASES,
        min_freq_word_unigram=20,
        top_k_word_unigram=TOP_K_WORD_UNIGRAM,
        min_freq_subword=50,
        top_k_subword=100_000,
    )
    print(f"[BUILD_VOCAB]: Initial vocab size: {len(vocab)}")

    # 5. Convert Corpus to Base-ID sequences for Unigram Training
    print("[BUILD_VOCAB]: Converting corpus to base-id sequences...")
    sequences: List[List[int]] = []
    for line in read_lines(TRAIN_PATH):
        if not line:
            continue
        base_seq = text_to_base_ids(line, char2id)
        sequences.append(base_seq)
    print(f"[BUILD_VOCAB]: Total Sequences: {len(sequences)}")

    # 6. Run Unigram Style Pruning
    final_vocab = train_unigram_vocab(
        sequences=sequences,
        vocab=vocab,
        target_size=TARGET_VOCAB_SIZE,
        prune_rate=PRUNE_RATE,
        max_iters=MAX_ITERS
    )

    print(f"[BUILD_VOCAB]: Final Vocab Size: {len(final_vocab)}")

    # 7. Export Vocab
    print(f"[BUILD_VOCAB]: Exporting vocab to {VOCAB_OUT_PATH}...")
    export_vocab(final_vocab, id2char, VOCAB_OUT_PATH)
    print("[BUILD_VOCAB]: DONE.")

if __name__ == "__main__":
    main()