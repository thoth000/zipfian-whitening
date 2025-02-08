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
from calc_isotropy_score import calc_uniform_isotropy_score, calc_zipfian_isotropy_score
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


def main(
    model_name: str = "sentence-transformers/average_word_embeddings_glove.840B.300d",
    topk=None,
):
    embedding_layer_index = 0
    pooling_layer_index = 1

    if model_name == "models/GoogleNews-vectors-negative300":
        model: SentenceTransformer = load_word2vec_model(model_name)
    else:
        model = SentenceTransformer(model_name)

    # To match the experiment setting to the original paper
    model.tokenizer.stop_words = {}
    model.tokenizer.do_lower_case = True
    model.tokenizer = WrappedTokenizer(model.tokenizer)
    model_vocab_size = model[embedding_layer_index].emb_layer.weight.shape[0]
    unigramprob: UnigramProb = load_unigram_prob_enwiki_vocab_min200(
        model.tokenizer, model_vocab_size
    )
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
    embedding_for_whitening, unigramprob_tensor = remove_unused_words(
        unsued_vocab_ids,
        model[embedding_layer_index].emb_layer.weight,
        unigramprob_tensor,
    )
    if topk is not None:
        idx = torch.argsort(unigramprob_tensor, descending=True)
        unigramprob_tensor = unigramprob_tensor[idx[:topk]]
        unigramprob_tensor /= unigramprob_tensor.sum()
        embedding_for_whitening = embedding_for_whitening[idx[:topk]]
    results = {
        "prob": unigramprob_tensor.tolist(),
    }

    ### Normal
    W_norm = torch.norm(embedding_for_whitening, dim=1).tolist()
    results["norm"] = W_norm

    ### Uniform
    uniform_whitening = UniformWhitening().fit(embedding_for_whitening, None)
    ## Centering only
    W_uniform_centered = embedding_for_whitening - uniform_whitening.mu
    W_uniform_centered_norm = torch.norm(W_uniform_centered, dim=1).tolist()
    sim1_uniform = calc_uniform_isotropy_score(W_uniform_centered)[0]
    print(f"Uniform sim1: {sim1_uniform}")
    results["uniform_centered_norm"] = W_uniform_centered_norm
    # Whitening
    W_uniform_whitened = W_uniform_centered @ uniform_whitening.transformation_matrix
    W_uniform_whitened_norm = torch.norm(W_uniform_whitened, dim=1).tolist()
    results["uniform_whitened_norm"] = W_uniform_whitened_norm
    sim2_uniform = calc_uniform_isotropy_score(W_uniform_whitened)[1]
    print(f"Uniform sim2: {sim2_uniform}")
    ### Zipfian
    zipfian_whitening = ZipfianWhitening().fit(
        embedding_for_whitening, unigramprob_tensor
    )
    # Centering only
    W_zipfian_centered = embedding_for_whitening - zipfian_whitening.mu
    W_zipfian_centered_norm = torch.norm(W_zipfian_centered, dim=1).tolist()
    results["zipfian_centered_norm"] = W_zipfian_centered_norm
    sim1_zipfian = calc_zipfian_isotropy_score(W_zipfian_centered, unigramprob_tensor)[
        0
    ]
    print(f"Zipfian sim1: {sim1_zipfian}")
    # Whitening
    W_zipfian_whitened = W_zipfian_centered @ zipfian_whitening.transformation_matrix
    W_zipfian_whitened_norm = torch.norm(W_zipfian_whitened, dim=1).tolist()
    sim2_zipfian = calc_zipfian_isotropy_score(W_zipfian_whitened, unigramprob_tensor)[
        1
    ]
    print(f"Zipfian sim2: {sim2_zipfian}")
    results["zipfian_whitened_norm"] = W_zipfian_whitened_norm

    # Dump results to json
    model_name = Path(model_name).name  # remove "SentenceTransformer/" prefix
    if topk is None:
        results_path = (
            Path("results") / model_name / "norm_experiments" / "results.json"
        )
    else:
        results_path = (
            Path("results")
            / model_name
            / "norm_experiments"
            / str(topk)
            / "results.json"
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    Fire(main)
