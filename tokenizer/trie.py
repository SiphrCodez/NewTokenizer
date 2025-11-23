from typing import Dict, Optional, Tuple, List

class TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.token_id: Optional[int] = None

class PieceTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, piece: List[int], token_id: int):
        node = self.root
        for bid in piece:
            if bid not in node.children:
                node.children[bid] = TrieNode()
            node = node.children[bid]
        node.token_id = token_id
    
    def longest_match(self, base_seq: List[int], start: int) -> Tuple[Optional[int], int]:
        """
        Returns (token_id, length_in_base_ids)
        If no match, returns (None, 0)
        """
        node = self.root
        best_token_id: Optional[int] = None
        best_len = 0

        i = start
        n = len(base_seq)

        while i < n:
            bid = base_seq[i]
            if bid not in node.children:
                break
            node = node.children[bid]
            i += 1
            if node.token_id is not None:
                best_token_id = node.token_id
                best_len = i - start
        
        if best_token_id is None:
            return None, 0
        return best_token_id, best_len
    
def build_trie_from_vocab(vocab: Dict[int, "Token"]) -> PieceTrie:
    from .vocab_builder import Token # Avoid circular import

    trie = PieceTrie()
    for token_id, token in vocab.items():
        if token.type == "BASE":
            trie.insert(token.piece, token_id)
        else:
            trie.insert(token.piece, token_id)
    return trie