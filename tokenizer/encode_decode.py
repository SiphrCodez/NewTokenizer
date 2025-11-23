from typing import List, Dict, TYPE_CHECKING
from .base_alphabet import SPECIAL_TOKENS
from .trie import PieceTrie

if TYPE_CHECKING:
    from .vocab_builder import Token

def text_to_base_ids(text: str, char2id: Dict[str, int]) -> List[int]:
    base_ids = []
    unk_id = SPECIAL_TOKENS["<UNK>"]
    for ch in text:
        base_ids.append(char2id.get(ch, unk_id))
    return base_ids

def base_ids_to_text(base_ids: List[int], id2char: Dict[int, str]) -> str:
    chars = []
    for bid in base_ids:
        ch = id2char.get(bid, "") # Skip unknown characters
        chars.append(ch)
    return "".join(chars)

def encode(text: str, char2id: Dict[str, int], trie: PieceTrie, base_char_token_map: Dict[int, int]) -> List[int]:
    base_seq = text_to_base_ids(text, char2id)
    tokens: List[int] = []

    i = 0
    n = len(base_seq)
    while i < n:
        token_id, length = trie.longest_match(base_seq, i)
        if token_id is None:
            # Fallback to single base character token
            base_id = base_seq[i]
            token_id = base_char_token_map.get(base_id, SPECIAL_TOKENS["<UNK>"])
            length = 1
        tokens.append(token_id)
        i += length
    return tokens

def decode(token_ids: List[int], vocab: Dict[int, 'Token'], id2char: Dict[int, str]) -> str:
    base_seq: List[int] = []
    for tid in token_ids:
        token = vocab.get(tid)
        if token is None:
            continue # Skip unknown tokens
        base_seq.extend(token.piece)
    return base_ids_to_text(base_seq, id2char)