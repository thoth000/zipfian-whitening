"""
Pre calculate the norm of uniform centering / whitening embeddings
to later use in the norm/direction experiments.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import sentence_transformers
import torch
from fire import Fire
from IPython import embed
from mteb import MTEB
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, WordEmbeddings
from torchtyping import TensorType as TT

from all_but_the_top import AllButTheTop
from eval_utils import (
    UnigramProb,
    WrappedTokenizer,
    load_unigram_prob_enwiki_vocab_min200,
    load_word2vec_model,
    remove_unused_words,
)
from modeling import CustomPooling
from SIF import SIF
from zipfian_whitening import UniformWhitening, ZipfianWhitening

# Downloaded from: https://github.com/kawine/usif/raw/71ffef5b6d7295c36354136bfc6728a10bd25d32/enwiki_vocab_min200.txt
PATH_ENWIKI_VOCAB_MIN200 = "data/enwiki_vocab_min200/enwiki vocab min200.txt"
TRANSFORM_CONFIG = {
    "normal": {
        "whitening_transformer_class": None,
        "pooling": ["mean"],
    },  # no whitening. normal mean pooling.
    "uniform_whitening": {
        "whitening_transformer_class": UniformWhitening,
        "pooling": ["centering_only", "whitening"],
    },
    "zipfian_whitening": {
        "whitening_transformer_class": ZipfianWhitening,
        "pooling": ["centering_only", "whitening"],
    },
    "abtp": {
        "whitening_transformer_class": AllButTheTop,
        "pooling": ["component_removal"],
    },
}


def main(model_name: str, topk: Optional[int] = None) -> None:
    print(f"topk: {topk}")
    embedding_layer_index = 0
    pooling_layer_index = 1

    if model_name == "models/GoogleNews-vectors-negative300":
        model: SentenceTransformer = load_word2vec_model(
            model_name, from_text_file=True
        )
    elif (
        model_name == "models/GoogleNews-vectors-negative300-torch"
        or model_name == "models/fasttext-ja-torch"
        or model_name == "models/fasttext-en-torch"
        or model_name == "models/fasttext-en-subword-torch"
    ):
        model: SentenceTransformer = load_word2vec_model(
            model_name, from_text_file=False
        )
    else:
        model = SentenceTransformer(model_name)

    model.tokenizer.stop_words = {}
    model.tokenizer.do_lower_case = True
    model.tokenizer = WrappedTokenizer(model.tokenizer)
    model_vocab_size = model[embedding_layer_index].emb_layer.weight.shape[0]
    unigramprob: UnigramProb = load_unigram_prob_enwiki_vocab_min200(
        model.tokenizer, model_vocab_size, topk=topk
    )
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
    params = {
        "model": model,
        "model_name": model_name,
        "whitening_transformer": None,
        "embedding_layer_index": embedding_layer_index,
        "pooling_layer_index": pooling_layer_index,
        "topk": topk,
    }
    embedding_for_whitening, unigramprob_tensor = remove_unused_words(
        unsued_vocab_ids,
        model[embedding_layer_index].emb_layer.weight,
        unigramprob_tensor,
    )
    model.tokenizer.original_tokenizer.stop_words = {
        model.tokenizer.vocab[index] for index in unsued_vocab_ids
    }

    # Fit Uniform whitening
    whitening_transformer = UniformWhitening().fit(embedding_for_whitening, p=None)

    # Apply Uniform whitening to word embeddings
    original_word_embeddings = model[embedding_layer_index].emb_layer.weight

    # 1. Centering
    original_word_embeddings -= whitening_transformer.mu
    norm = torch.linalg.norm(original_word_embeddings, dim=1)
    save_path = f"data/norm/uniform/centering/{model_name.split('/')[-1]}.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(norm, save_path)

    # 2. Whitening
    original_word_embeddings = (
        original_word_embeddings @ whitening_transformer.transformation_matrix
    )
    norm = torch.linalg.norm(original_word_embeddings, dim=1)
    save_path = f"data/norm/uniform/whitening/{model_name.split('/')[-1]}.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(norm, save_path)


if __name__ == "__main__":
    Fire(main)
