from typing import Dict, Tuple

def build_base_alphabet() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a base alphabet mapping for tokenization.

    Returns:
    - char2id: char -> base_id (ints)
    - id2char: base_id (ints) -> char
    """
    # Reserve a few special tokens (not included in the base alphabet)
    # 0: <pad>, 1: <unk>, 2: <bos>, 3: <eos>
    next_id = 4

    char2id = {}
    id2char = {}

    # Minimal Version: ASCII Letters, Digits, Punctuation, Space, Newline
    alphabet = []
    alphabet.extend([chr(c) for c in range(32,127)]) # ASCII printable characters

    for char in alphabet:
        char2id[char] = next_id
        id2char[next_id] = char
        next_id += 1

    if "\n" not in char2id:
        char2id["\n"] = next_id
        id2char[next_id] = "\n"
        next_id += 1

    return char2id, id2char

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}