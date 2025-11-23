from tokenizer.ngram_extraction import extract_ngram_stats
import json

def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line

if __name__ == "__main__":
    lines = read_lines("data/osb_raw.txt")
    (char_ngram_freq, word_ngram_freq, unigram_word_freq, context_left, context_right) = extract_ngram_stats(lines, max_char_n=6, max_word_n=5, min_char_len=2, min_word_n=2)

    with open("artifacts/ngram_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "char_ngram_freq_len": len(char_ngram_freq),
            "word_ngram_freq_len": len(word_ngram_freq),
            "unigram_word_freq_len": len(unigram_word_freq)
        }, f)