import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union

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
    WrappedTokenizerJA,
    WrappedTokenizerSubword,
    load_fasttext_model,
    load_unigram_prob_enwiki_vocab_min200,
    load_unigram_prob_in_batch,
    load_unigram_prob_jawiki_top_20k,
    load_word2vec_model,
    remove_unused_words,
)
from modeling import CustomPooling
from SIF import SIF
from zipfian_whitening import UniformWhitening, ZipfianWhitening

TRANSFORM_CONFIG = {
    "normal": {
        "whitening_transformer_class": None,
        "pooling": [
            "mean",  # no whitening. normal mean pooling.
            "raw_then_zipfian_whitening_norm",
        ],
    },
    "uniform_whitening": {
        "whitening_transformer_class": UniformWhitening,
        "pooling": [
            "centering_only",
            "whitening",
            "uniform_centering_then_zipfian_whitening_norm",
            "uniform_whitening_then_zipfian_whitening_norm",
        ],
    },
    "zipfian_whitening": {
        "whitening_transformer_class": ZipfianWhitening,
        "pooling": [
            "centering_only",
            "whitening",
            "raw_then_zipfian_whitening_dirction",
            "zipfian_whitening_then_uniform_centering_norm",
            "zipfian_whitening_then_uniform_whitening_norm",
        ],
    },
    "abtp": {
        "whitening_transformer_class": AllButTheTop,
        "pooling": ["component_removal"],
    },
}
MODEL_NAME_TO_WRAPPED_TOKENIZER = {
    "models/GoogleNews-vectors-negative300-torch": WrappedTokenizer,
    "sentence-transformers/average_word_embeddings_glove.840B.300d": WrappedTokenizer,
    "models/fasttext-ja-torch": WrappedTokenizerJA,
    "models/fasttext-en-torch": WrappedTokenizer,
    "models/fasttext-en-subword-torch": WrappedTokenizerSubword,
}
MODEL_NAME_TO_FREQ_FUNC = {
    "models/GoogleNews-vectors-negative300-torch": load_unigram_prob_enwiki_vocab_min200,
    "models/fasttext-ja-torch": load_unigram_prob_jawiki_top_20k,
    "models/fasttext-en-torch": load_unigram_prob_enwiki_vocab_min200,
    "models/fasttext-en-subword-torch": load_unigram_prob_enwiki_vocab_min200,
    "sentence-transformers/average_word_embeddings_glove.840B.300d": load_unigram_prob_enwiki_vocab_min200,
}
MODEL_NAME_TO_FREQ_FUNC_IN_BATCH = {
    "models/GoogleNews-vectors-negative300-torch": load_unigram_prob_in_batch,
    "models/fasttext-ja-torch": load_unigram_prob_in_batch,
    "models/fasttext-en-torch": load_unigram_prob_in_batch,
    "models/fasttext-en-subword-torch": load_unigram_prob_in_batch,
    "sentence-transformers/average_word_embeddings_glove.840B.300d": load_unigram_prob_in_batch,
}

TASK_NAME_TO_SPLIT_NAME = {
    "STSBenchmark": "test",
    "SICK-R": "test",
    "STS12": "test",
    "STS13": "test",
    "STS14": "test",
    "STS15": "test",
    "STS16": "test",
    "JSTS": "validation",  # XXX: eval split is set on validation in MTEB
}


def evaluate(
    model: SentenceTransformer,
    model_name: str,
    task_name: str,
    whitening_transformer: Union[UniformWhitening, ZipfianWhitening],
    pooling_mode: str,
    embedding_layer_index: int = 0,
    pooling_layer_index: int = 1,
    topk: Optional[int] = None,
    in_batch: bool = False,
):
    # save_dir name rule:
    # {model_name}/{task_name}/{whitening_transformer_name (e.g., zipfian_whitening, ...)}/{pooling_mode}/{topk}
    model_name = Path(model_name).name  # remove "SentenceTransformer/" prefix
    if in_batch:
        model_name = f"{model_name}_in_batch"
    if whitening_transformer is None:
        whitening_name = "normal"
    elif isinstance(whitening_transformer, ZipfianWhitening):
        whitening_name = "zipfian_whitening"
    elif isinstance(whitening_transformer, UniformWhitening):
        whitening_name = "uniform_whitening"
    elif isinstance(whitening_transformer, AllButTheTop):
        whitening_name = "abtp"
    elif isinstance(whitening_transformer, SIF):
        whitening_name = "sif"
    else:
        raise NotImplementedError(
            'Only "ZipfianWhitening" and "UniformWhitening" and "AllButTheTop" and "SIF" are supported.'
        )
    save_dir_name = (
        f"results/{model_name}/{task_name}/{whitening_name}/{pooling_mode}"
        if topk is None
        else f"results/{model_name}/{task_name}/{whitening_name}/{pooling_mode}/{topk}"
    )
    pooling = CustomPooling(
        word_embedding_dimension=model[
            embedding_layer_index
        ].get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
        whitening_transformer=whitening_transformer,
    )
    model[pooling_layer_index] = pooling
    task = MTEB(tasks=[task_name])
    results = task.run(
        model,
        output_folder=save_dir_name,
    )
    print("#" * 50)
    print(f"Done {model_name} with {whitening_name} and {pooling_mode} pooling.")
    print("#" * 50)
    pprint(results)


def main(
    model_name: str,
    task_names: List[str] = [
        "STSBenchmark",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
    ],
    topk: Optional[int] = None,
    in_batch: bool = False,
) -> None:
    print(f"topk: {topk}")
    print(type(task_names), task_names)

    # Load model
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
    elif model_name == "models/crawl-300d-2M.vec":
        model: SentenceTransformer = load_fasttext_model(model_name)
    else:
        model = SentenceTransformer(model_name)

    # Setup tokenizer
    embedding_layer_index = 0
    pooling_layer_index = 1
    model.tokenizer.stop_words = {}
    model.tokenizer.do_lower_case = True
    wrapped_tokenizer_class = MODEL_NAME_TO_WRAPPED_TOKENIZER.get(model_name)
    model.tokenizer = wrapped_tokenizer_class(model.tokenizer)
    model_vocab_size = model[embedding_layer_index].emb_layer.weight.shape[0]

    # Load unigram probabilities
    unigramprob_func = MODEL_NAME_TO_FREQ_FUNC.get(model_name)
    unigramprob: UnigramProb = unigramprob_func(
        model.tokenizer, model_vocab_size, topk=topk
    )
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids

    # To reduce the noise for the whitening, remove the unused words from the embedding and unigram probabilities.
    # This is common setting for all the whitening methods.
    embedding_for_whitening, unigramprob_tensor = remove_unused_words(
        unsued_vocab_ids,
        model[embedding_layer_index].emb_layer.weight,
        unigramprob_tensor,
    )
    model.tokenizer.original_tokenizer.stop_words = {  # HACK: Setting unused words (i.e., w with p(w)=0) as stop words. The model ignores these tokens.
        model.tokenizer.vocab[index] for index in unsued_vocab_ids
    }

    params = {
        "model": model,
        "model_name": model_name,
        "whitening_transformer": None,
        "embedding_layer_index": embedding_layer_index,
        "pooling_layer_index": pooling_layer_index,
        "topk": topk,
    }

    ############################
    # Evaluate non-SIF methods #
    ############################
    for task_name in task_names:
        params["task_name"] = task_name
        if in_batch:
            # In batch setting (test time frequency setting).
            model.tokenizer = model.tokenizer.original_tokenizer
            model.tokenizer.stop_words = {}
            model.tokenizer.do_lower_case = True
            wrapped_tokenizer_class = MODEL_NAME_TO_WRAPPED_TOKENIZER.get(model_name)
            model.tokenizer = wrapped_tokenizer_class(model.tokenizer)
            unigramprob_func = MODEL_NAME_TO_FREQ_FUNC_IN_BATCH.get(model_name)
            unigramprob: UnigramProb = (
                unigramprob_func(  # Get test time frequency of the task
                    model.tokenizer,
                    model_vocab_size,
                    task_name,
                    TASK_NAME_TO_SPLIT_NAME[task_name],
                    topk=topk,
                )
            )
            unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
            unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
            embedding_for_whitening, unigramprob_tensor = remove_unused_words(
                unsued_vocab_ids,
                model[embedding_layer_index].emb_layer.weight,
                unigramprob_tensor,
            )
            model.tokenizer.original_tokenizer.stop_words = {  # HACK: Setting unused words (i.e., w with p(w)=0) as stop words. The model ignores these tokens.
                model.tokenizer.vocab[index] for index in unsued_vocab_ids
            }
            params = {
                "model": model,
                "model_name": model_name,
                "whitening_transformer": None,
                "embedding_layer_index": embedding_layer_index,
                "pooling_layer_index": pooling_layer_index,
                "topk": topk,
                "in_batch": in_batch,
                "task_name": task_name,
            }
        for trnaform_name in TRANSFORM_CONFIG:
            params["pooling_mode"]: List[str] = TRANSFORM_CONFIG[trnaform_name][
                "pooling"
            ]
            whitening_transformer = TRANSFORM_CONFIG[trnaform_name][
                "whitening_transformer_class"
            ]
            whitening_transformer = (
                None
                if whitening_transformer is None
                else whitening_transformer().fit(
                    embedding_for_whitening, p=unigramprob_tensor
                )
            )
            params["whitening_transformer"] = whitening_transformer
            for pooling_mode in params["pooling_mode"]:
                params["pooling_mode"] = pooling_mode
                evaluate(**params)

    ################
    # Evaluate SIF #
    ################
    unigramprob_func = MODEL_NAME_TO_FREQ_FUNC.get(model_name)
    unigramprob: UnigramProb = unigramprob_func(
        model.tokenizer, model_vocab_size, topk=topk
    )
    unsued_vocab_ids: set[int] = unigramprob.unused_vocab_ids
    unigramprob_tensor: TT["num_words"] = unigramprob.prob.to(model.device)
    unigramprob_tensor[list(unsued_vocab_ids)] = 0
    unigramprob_tensor = unigramprob_tensor / unigramprob_tensor.sum()
    model.tokenizer.original_tokenizer.stop_words = {  # HACK: Setting unused words (i.e., w with p(w)=0) as stop words. The model ignores these tokens.
        model.tokenizer.vocab[index] for index in unsued_vocab_ids
    }
    for task_name in task_names:
        sif = SIF(
            model, task_name=task_name, data_split=TASK_NAME_TO_SPLIT_NAME[task_name]
        )
        sif.fit(None, unigramprob_tensor)
        pooling_mode = "sif_w_component_removal"
        evaluate(
            model=model,
            model_name=model_name,
            task_name=task_name,
            whitening_transformer=sif,
            pooling_mode=pooling_mode,
            embedding_layer_index=embedding_layer_index,
            pooling_layer_index=pooling_layer_index,
            topk=topk,
        )


if __name__ == "__main__":
    Fire(main)
