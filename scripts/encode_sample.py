from tokenizer.base_alphabet import build_base_alphabet
from tokenizer.trie import build_trie_from_vocab
from tokenizer.encode_decode import encode, decode
from tokenizer.vocab_builder import Token
import json

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
                surface=obj.get("surface", "")
            )
            vocab[token.id] = token
    return vocab

if __name__ == "__main__":
    # Build mappings
    char2id, id2char = build_base_alphabet()
    vocab = load_vocab("artifacts/vocab_v1.jsonl")
    trie = build_trie_from_vocab(vocab)

    base_char_token_map = {}
    for t in vocab.values():
        if t.type == "BASE" and len(t.piece) == 1:
            base_char_token_map[t.piece[0]] = t.id
    
    sample_text = "In the mighty name of the Most High, the Alpha and Omega, the beginning and the end, the first and the last."
    token_ids = encode(sample_text, char2id, trie, base_char_token_map)

    print("Text: ", sample_text)
    print("Tokens: ", token_ids)
    print("# of Tokens: ", len(token_ids))

    recon = decode(token_ids, vocab, id2char)
    print("Decoded: ", recon)