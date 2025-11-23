from tokenizer.base_alphabet import build_base_alphabet
import json

if __name__ == "__main__":
    char2id, id2char = build_base_alphabet()
    with open("artifacts/base_alphabet.json", "w", encoding="utf-8") as f:
        json.dump({"char2id": char2id, "id2char": id2char}, f, ensure_ascii=False)