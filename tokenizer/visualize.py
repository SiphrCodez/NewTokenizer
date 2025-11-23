from typing import Dict, List
from .vocab_builder import Token

def print_top_tokens_by_freq(vocab: Dict[int, Token], top_k: int = 50):
    tokens = sorted(
        [t for t in vocab.values() if t.type != "BASE"],
        key=lambda t: t.freq,
        reverse=True,
    )
    for t in tokens[:top_k]:
        print(f"[{t.id}] ({t.type}) freq={t.freq:.0f}  surface='{t.surface}'")

def print_long_phrase_tokens(vocab: Dict[int, Token], min_chars: int = 15, top_k: int = 50):
    candidates = List[Token] = []
    for t in vocab.values():
        if t.type in ("WORD", "PHRASE") and len(t.surface) >= min_chars:
            candidates.append(t)
    candidates.sort(key=lambda t: len(t.surface), reverse=True)
    for t in candidates[:top_k]:
        print(f"[{t.id}] len={len(t.surface)} freq={t.freq:.0f} '{t.surface}'")