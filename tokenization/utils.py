from typing import Dict, List
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import json
import os

def build_asin_id_tokenizer(asin2id: Dict[str, int]) -> PreTrainedTokenizerFast:
    """
    Create a tokenizer where tokens are the *stringified numeric ids* from asin2id.values(),
    plus special tokens for generative modeling.
    """
    # Base vocab: map stringified id -> original id
    base_ids = list(asin2id.values())
    max_id = max(base_ids)
    vocab = {str(v): v for v in base_ids}

    # Reserve special tokens *after* the max existing id (no clashes)
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    special_to_id = {
        "<pad>": max_id + 1,
        "<bos>": max_id + 2,
        "<eos>": max_id + 3,
        "<unk>": max_id + 4,
        " " :  max_id + 5,
    }
    vocab.update(special_to_id)

    # Low-level tokenizer
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()  # expects space-separated tokens like "12 45 7"

    # Wrap as HF fast tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=4096,
    )

    # Set ids explicitly (helps some models/trainers)
    hf_tok._tokenizer.model = tok.model  # ensure shared model
    hf_tok.bos_token_id = special_to_id["<bos>"]
    hf_tok.eos_token_id = special_to_id["<eos>"]
    hf_tok.unk_token_id = special_to_id["<unk>"]
    hf_tok.pad_token_id = special_to_id["<pad>"]

    # Add convenience helpers (optional)
    def encode_id_sequence(id_seq: List[int], add_special_tokens: bool = True):
        text = " ".join(str(i) for i in id_seq)
        return hf_tok(text, add_special_tokens=add_special_tokens)

    def decode_id_sequence(input_ids: List[int], skip_special_tokens: bool = True):
        text = hf_tok.decode(input_ids, skip_special_tokens=skip_special_tokens)
        # Turn "12 45 7" back into [12, 45, 7]
        return [int(t) for t in text.split()] if text.strip() else []

    hf_tok.encode_id_sequence = encode_id_sequence  # type: ignore
    hf_tok.decode_id_sequence = decode_id_sequence  # type: ignore

    return hf_tok

def save_tokenizer(hf_tok: PreTrainedTokenizerFast, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hf_tok.save_pretrained(out_dir)

def load_tokenizer(out_dir: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(out_dir)