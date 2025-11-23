import json
from tokenizer.visualize import print_top_tokens_by_freq, print_long_phrase_tokens
from tokenizer.vocab_builder import Token

def load_vocab(path: str):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            token = Token(
                id=obj["id"],
                piece=obj["piece"],
                type=obj["type"],
                freq=obj.get("freq", 0.0),
                score=obj.get("score", 0.0),
                surface=obj.get("surface", ""),
            )
            vocab[token.id] = token
    return vocab

if __name__ == "__main__":
    vocab = load_vocab("artifacts/vocab_v1.jsonl")
    print("=== Top Tokens by Frequency ===")
    print_top_tokens_by_freq(vocab, top_k=50)

    print("\n=== Long Phrase Tokens (min 15 chars) ===")
    print_long_phrase_tokens(vocab, min_chars=15, top_k=50)