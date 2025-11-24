from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple
import json
import math
import re

from .base_alphabet import SPECIAL_TOKENS
from .trie import build_trie_from_vocab
from .encode_decode import text_to_base_ids

@dataclass
class Token:
    id: int
    piece: List[int] # Sequence of base character IDs
    type: str # e.g., "BASE", "SUBWORD", "WORD", "PHRASE"
    freq: float = 0.0
    score: float = 0.0
    surface: str = "" # Human readable form of the token

WORD_CHARS_RE = re.compile(r"\w", re.UNICODE)


def add_word_unigram_tokens(
    vocab: Dict[int, Token],
    next_id: int,
    unigram_word_freq: Dict[str, int],
    char2id: Dict[str, int],
    min_freq_word: int = 20,
    top_k_word_unigram: int = 50_000,
    add_leading_space_variant: bool = True
) -> int:
    """
    Add single-word tokens (WORD type) based on frequency.

    For each high-frequency word:
      - Always add a plain token: "word"
      - Optionally also add a leading-space variant: " word"

    This allows mid-sentence occurrences to be encoded as one token
    (space+word) instead of " " + "word".
    """

    # Sort words by frequency (descending)
    sorted_words = sorted(
        unigram_word_freq.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    added = 0
    for word, freq in sorted_words:
        if freq < min_freq_word:
            break
        if added >= top_k_word_unigram:
            break

        # Simple filter to avoid weird "words" that are pure punctuation etc.
        # You can relax this if needed.
        if not any(WORD_CHARS_RE.search(ch) for ch in word):
            continue

        piece = ngram_to_piece(word, char2id)
        if piece is None:
            continue

        plain_token = Token(
            id=next_id,
            piece=piece,
            type="WORD",
            freq=float(freq),   # store initial freq, useful for later inspection
            score=0.0,
            surface=word,
        )
        vocab[next_id] = plain_token
        next_id += 1
        added += 1

        # Leading Space Variant: " word"
        if add_leading_space_variant and " " in char2id:
            spaced_surface = " " + word
            spaced_piece = ngram_to_piece(spaced_surface, char2id)
            if spaced_piece is not None:
                spaced_token = Token(
                    id=next_id,
                    piece=spaced_piece,
                    type="WORD",
                    freq=float(freq),
                    score=0.0,
                    surface=spaced_surface,
                )
                vocab[next_id] = spaced_token
                next_id += 1
            spaced_surface = " " + word + " "
            spaced_piece = ngram_to_piece(spaced_surface, char2id)
            if spaced_piece is not None:
                spaced_token = Token(
                    id=next_id,
                    piece=spaced_piece,
                    type="WORD",
                    freq=float(freq),
                    score=0.0,
                    surface=spaced_surface
                )
                vocab[next_id] = spaced_token
                next_id += 1
            spaced_surface = word + " "
            spaced_piece = ngram_to_piece(spaced_surface, char2id)
            if spaced_piece is not None:
                spaced_token = Token(
                    id=next_id,
                    piece=spaced_piece,
                    type="WORD",
                    freq=float(freq),
                    score=0.0,
                    surface=spaced_surface,
                )
                vocab[next_id] = spaced_token
                next_id += 1

    print(f"[VOCAB_BUILDER] Added {added} word-unigram tokens.")
    return next_id

def add_special_tokens(vocab: Dict[int, Token]) -> int:
    """
    Initialize SPECIAL tokens (PAD, UNK, BOS, EOS) into vocab.
    Returns next available token_id.
    """
    next_id = 0
    for name, tid in SPECIAL_TOKENS.items():
        token = Token(
            id=tid,
            piece=[],
            type="SPECIAL",
            freq=0.0,
            score=0.0,
            surface=name,
        )
        vocab[tid] = token
        next_id = max(next_id, tid + 1)
    return next_id

def add_base_char_tokens(
    vocab: Dict[int, Token], 
    next_id: int,
    char2id: Dict[str, int],
    id2char: Dict[int, str],
) -> int:
    """
    Add a Token for every base character (char2id entry)
    Each base token's piece is a single base_id
    Returns next available token_id
    """
    for base_id in sorted(id2char.keys()):
        ch = id2char[base_id]
        token = Token(
            id=next_id,
            piece=[base_id],
            type="BASE",
            freq=0.0,
            score=0.0,
            surface=ch
        )
        vocab[next_id] = token
        next_id += 1
    return next_id

def ngram_to_piece(ngram: str, char2id: Dict[str, int]) -> List[int] | None:
    """
    Convert ngram string into a sequence of base_ids.
    Returns None if any character can't be mapped.
    """
    piece: List[int] = []
    for ch in ngram:
        if ch not in char2id:
            return None
        piece.append(char2id[ch])
    return piece

def classify_ngram_type(ngram: str) -> str:
    """
    Rough classifcation of ngram type based on whether it has spaces.
    *** REFINE THIS LATER ***
    """
    if " " in ngram:
        num_words = len(ngram.split())
        if num_words == 1:
            return "WORD"
        else:
            return "PHRASE"
    else:
        return "SUBWORD"
    
def build_initial_vocab_from_ngrams(
    char2id: Dict[str, int],
    id2char: Dict[int, str],
    scored_word_ngrams: Dict[str, float],
    unigram_word_freq: Dict[str, int],
    top_k_phrases: int = 300_000,
    min_freq_word_unigram: int = 20,
    top_k_word_unigram: int = 50_000,
) -> Dict[int, Token]:
    """
    Create initial vocabulary:
    - 1. SPECIAL Tokens.
    - 2. BASE Char Tokens.
    - 3. Top-K word ngram tokens (multi-word phrases)
    - 4. High Frequency Word Unigrams
    Later also add char ngrams here if you want SUBWORD candidates.
    """
    vocab: Dict[int, Token] = {}
    # 1. Special Tokens
    next_id = add_special_tokens(vocab)

    # 2. Base Character Tokens
    next_id = add_base_char_tokens(vocab, next_id, char2id, id2char)

    # 3. Word Unigrams (single words) based on unigram_word_freq
    next_id = add_word_unigram_tokens(
        vocab=vocab,
        next_id=next_id,
        unigram_word_freq=unigram_word_freq,
        char2id=char2id,
        min_freq_word=min_freq_word_unigram,
        top_k_word_unigram=top_k_word_unigram
    )

    # 4. Multi-word Phrase Tokens from scored ngrams
    sorted_ngrams = sorted(
        scored_word_ngrams.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    added_phrases = 0
    for ngram, score in sorted_ngrams:
        if added_phrases >= top_k_phrases:
            break
        # Only keep multi-word ngrams here
        if " " not in ngram:
            continue
        piece = ngram_to_piece(ngram, char2id)
        if piece is None:
            continue

        token = Token(
            id=next_id,
            piece=piece,
            type="PHRASE",
            freq=0.0,
            score=score,
            surface=ngram
        )
        vocab[next_id] = token
        next_id += 1
        added_phrases += 1

    return vocab

def greedy_segment_sequence(
    base_seq: List[int],
    vocab: Dict[int, Token]
) -> List[int]:
    """
    Segment single sequence of base_ids into token_ids using current vocab
    and a greedy longest-match strategy.
    """
    from .trie import PieceTrie

    trie = build_trie_from_vocab(vocab)

    # Build base_id -> base_token_id map.
    base_char_token_map: Dict[int, int] = {}
    for t in vocab.values():
        if t.type == "BASE" and len(t.piece) == 1:
            base_char_token_map[t.piece[0]] = t.id

    token_ids: List[int] = []
    i = 0
    n = len(base_seq)
    while i < n:
        token_id, length = trie.longest_match(base_seq, i)
        if token_id is None or length == 0:
            # Fallback to base character token.
            bid = base_seq[i]
            fallback = base_char_token_map.get(
                bid, SPECIAL_TOKENS["<UNK>"]
            )
            token_ids.append(fallback)
            i += 1
        else:
            token_ids.append(token_id)
            i += length
    return token_ids

def train_unigram_vocab(
    sequences: Iterable[List[int]],
    vocab: Dict[int, Token],
    target_size: int,
    prune_rate: float = 0.1,
    min_freq_to_keep: int = 1,
    max_iters: int = 100,
) -> Dict[int, Token]:
    """
    Very simplified Unigram LM-style pruning.
    - Repeatedly:
        * Segment corpus with current vocab.
        * Count token frequencies.
        * Computer negative log-probs.
        * Prunt worst-scoring tokens (except BASE + SPECIAL)
    - Stop once vocab <= target_size or max_iters reached.

    This is not a full EM, only used for V1 experiments.
    """
    vocab = dict(vocab) # work on a copy of the vocabulary.

    def is_removable(t: Token) -> bool:
        return t.type not in ("BASE", "SPECIAL")
    
    it = 0
    while len(vocab) > target_size and it < max_iters:
        it += 1
        print(f"[UNIGRAM] Iteration {it}, vocab size = {len(vocab)}")

        # 1. Reset Freq
        for t in vocab.values():
            t.freq = 0.0
        
        # 2. Rebuild trie and segment all sequences.
        from .trie import PieceTrie
        trie = build_trie_from_vocab(vocab)

        # base_id -> base_token_id (for fallback purposes)
        base_char_token_map: Dict[int, int] = {}
        for t in vocab.values():
            if t.type == "BASE" and len(t.piece) == 1:
                base_char_token_map[t.piece[0]] = t.id
        
        for seq in sequences:
            # Greedy segmentation with current vocabulary.
            token_ids: List[int] = []
            i = 0
            n = len(seq)
            while i < n:
                token_id, length = trie.longest_match(seq, i)
                if token_id is None or length == 0:
                    bid = seq[i]
                    fallback = base_char_token_map.get(
                        bid, SPECIAL_TOKENS["<UNK>"]
                    )
                    token_ids.append(fallback)
                    i += 1
                else:
                    token_ids.append(token_id)
                    i += length
            
            # Accumulate Freq
            for tid in token_ids:
                vocab[tid].freq += 1.0
        
        # 3. Compute Scores (negative log prob)
        total_freq = sum(t.freq for t in vocab.values()) + 1e-9
        for t in vocab.values():
            if t.freq < min_freq_to_keep and is_removable(t):
                # Treat extremely rare tokens as very very bad.
                t.score = float("inf")
            else:
                p = (t.freq + 1e-9) / total_freq
                t.score = -math.log(p)

        # 4. Select Tokens to Prune
        removable = [t for t in vocab.values() if is_removable(t)]
        if not removable:
            print('[UNIGRAM]: No removable tokens remaining, breaking.')
            break
            
        removable.sort(key=lambda t: t.score, reverse=True) # Worst First
        num_to_prune = max(1, int(len(vocab) * prune_rate))
        num_to_prune = min(num_to_prune, len(removable) - 1)

        to_prune_ids = set(t.id for t in removable[:num_to_prune])
        for tid in to_prune_ids:
            del vocab[tid]
        
        print(
            f"[UNIGRAM]: Pruned {len(to_prune_ids)} tokens, "
            f"New Vocab Size: {len(vocab)}"
        )
    
    return vocab

def export_vocab(
    vocab: Dict[int, Token],
    id2char: Dict[int, str],
    path: str,
):
    """
    Export vocab as JSONL.
    Each line: {id, piece, type, freq, score, surface}
    """
    with open(path, "w", encoding="utf-8") as f:
        for token_id in sorted(vocab.keys()):
            t = vocab[token_id]
            obj = {
                "id": t.id,
                "piece": t.piece,
                "type": t.type,
                "freq": t.freq,
                "score": t.score,
                "surface": t.surface
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")