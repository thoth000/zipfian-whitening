import json
import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sentence_transformers
import torch
from fire import Fire
from IPython import embed
from mteb import MTEB
from nltk import word_tokenize
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, WordEmbeddings
from torchtyping import TensorType as TT
from tqdm import tqdm


import MeCab
import ipadic
from mteb.tasks import (
    STS12STS,
    STS13STS,
    STS14STS,
    STS15STS,
    STS16STS,
    STSBenchmarkSTS,
    SickrSTS,
    JSTS,
)
from tqdm import tqdm
import fasttext


PATH_ENWIKI_VOCAB_MIN200 = "data/enwiki_vocab_min200/enwiki vocab min200.txt"
PATH_JAWIKI_TOP_20K = "data/jawiki_top_20k/ja_wiki_top_20k.txt"

# TODO: to avoid de-duplication, move this to a common place (also used in SIF)
TASK_NAME_TO_DATASET = {
    "STS12": STS12STS,
    "STS13": STS13STS,
    "STS14": STS14STS,
    "STS15": STS15STS,
    "STS16": STS16STS,
    "STSBenchmark": STSBenchmarkSTS,
    "SICK-R": SickrSTS,
    "JSTS": JSTS,
}


@dataclass
class UnigramProb:
    """
    Dataclass to store the unigram probabilities of the words and the unused vocabulary ids. (will be used for removing the unused words from the embeddings before whitening)
    """

    prob: TT["model_vocab_size"]
    unused_vocab_ids: set[int]


class WrappedTokenizer(object):
    """
    Given a instanciated WhiteSpaceTokenizer, warp as a new tokenizer class to integrate nltk tokenizer
    """

    def __init__(
        self,
        original_tokenizer: sentence_transformers.models.tokenizer.WhitespaceTokenizer,
    ):
        self.original_tokenizer = original_tokenizer

    def tokenize(self, text: str) -> list[str]:
        """
        Override the tokenize method to use nltk_tokenize instead of the original whitespace_tokenize
        """
        text = self.nltk_tokenize(text)
        return self.original_tokenizer.tokenize(text)

    # Modified from https://github.com/kawine/usif/blob/71ffef5b6d7295c36354136bfc6728a10bd25d32/usif.py#L107-L137
    def nltk_tokenize(self, sentence: str) -> str:
        """
        Given a sentence, tokenize it using nltk's word_tokenize function.
        Then, preprocess the tokens by converting them to lowercase and removing punctuation.
        Finally, return the concatenated tokens as a string, ready to be passed to whitespace tokenization of the model.
        """
        # regex for non-punctuation
        not_punc = re.compile(".*[A-Za-z0-9].*")

        # preprocess a given token
        def preprocess(t):
            t = t.lower().strip("';.:()").strip('"')
            t = "not" if t == "n't" else t
            return re.split(r"[-]", t)

        tokens = []

        for token in word_tokenize(sentence):
            if not_punc.match(token):
                tokens = tokens + preprocess(token)

        return " ".join(tokens)

    def __getattr__(self, name):
        """
        Forward all other method calls to the original tokenizer
        """
        return getattr(self.original_tokenizer, name)


class WrappedTokenizerSubword(object):
    """
    Given a instanciated WhiteSpaceTokenizer, warp as a new tokenizer class to integrate nltk tokenizer
    """

    def __init__(
        self,
        original_tokenizer: sentence_transformers.models.tokenizer.WhitespaceTokenizer,
    ):
        fasttext_subword_model_path = (
            "/home/kurita/zipfian-whitening/models/crawl-300d-2M-subword.bin"
        )
        self.original_tokenizer = original_tokenizer
        print("loading fasttext subword model...")
        self.fasttext_subword_model = fasttext.load_model(fasttext_subword_model_path)
        print("done loading fasttext subword model")

    def tokenize(self, text: str) -> list[str]:
        """
        Override the tokenize method to use nltk_tokenize instead of the original whitespace_tokenize
        """
        text = self.nltk_tokenize(text)
        text = self.subword_tokenize(text)

        return self.original_tokenizer.tokenize(text)

    # Modified from https://github.com/kawine/usif/blob/71ffef5b6d7295c36354136bfc6728a10bd25d32/usif.py#L107-L137
    def nltk_tokenize(self, sentence: str) -> str:
        """
        Given a sentence, tokenize it using nltk's word_tokenize function.
        Then, preprocess the tokens by converting them to lowercase and removing punctuation.
        Finally, return the concatenated tokens as a string, ready to be passed to whitespace tokenization of the model.
        """
        # regex for non-punctuation
        not_punc = re.compile(".*[A-Za-z0-9].*")

        # preprocess a given token
        def preprocess(t):
            t = t.lower().strip("';.:()").strip('"')
            t = "not" if t == "n't" else t
            return re.split(r"[-]", t)

        tokens = []

        for token in word_tokenize(sentence):
            if not_punc.match(token):
                tokens = tokens + preprocess(token)

        return " ".join(tokens)

    def subword_tokenize(self, sentence: str) -> str:
        words = sentence.split()
        subwords = []
        for word in words:
            subwords += self.fasttext_subword_model.get_subwords(word)[
                0
            ]  # self.fasttext_subword_model.get_subwords(word)[0]
        return " ".join(subwords)

    def __getattr__(self, name):
        """
        Forward all other method calls to the original tokenizer
        """
        return getattr(self.original_tokenizer, name)


class WrappedTokenizerJA(object):
    """
    Given a instanciated WhiteSpaceTokenizer, warp as a new tokenizer class to integrate MeCab tokenizer.
    """

    def __init__(
        self,
        original_tokenizer: sentence_transformers.models.tokenizer.WhitespaceTokenizer,
    ):
        self.original_tokenizer = original_tokenizer
        self.mecab = MeCab.Tagger(ipadic.MECAB_ARGS)

    def tokenize(self, text: str) -> list[str]:
        """
        Override the tokenize method to use mecab_tokenize instead of the original whitespace_tokenize
        """
        text = self.mecab_tokenize(text)
        return self.original_tokenizer.tokenize(text)

    # Modified from https://github.com/kawine/usif/blob/71ffef5b6d7295c36354136bfc6728a10bd25d32/usif.py#L107-L137
    def mecab_tokenize(self, sentence: str) -> str:
        """
        Given a sentence, tokenize it using meacb word_tokenize function.
        Then, preprocess the tokens by removing punctuation.
        Finally, return the concatenated tokens as a string, ready to be passed to whitespace tokenization of the model.
        """
        # regex for non-punctuation
        not_punc = re.compile(".*[A-Za-z0-9].*")

        # preprocess a given token
        def preprocess(t):
            t = t.strip("';.:()").strip('"')
            return re.split(r"[-]", t)

        tokens = []

        tokenized = [
            text.split("\t")[0] for text in self.mecab.parse(sentence).split("\n")
        ]
        tokenized = tokenized[: tokenized.index("EOS")]

        for token in tokenized:
            if not_punc.match(token):
                tokens = tokens + preprocess(token)

        return " ".join(tokenized)

    def __getattr__(self, name):
        """
        Forward all other method calls to the original tokenizer
        """
        return getattr(self.original_tokenizer, name)


def load_unigram_prob(model_name: str, model_vocab_size: int) -> UnigramProb:
    unigram_prob_path = Path(
        f"data/wikipedia/{Path(model_name).name}/unigram_prob.json"
    )
    with unigram_prob_path.open("r") as f:
        unigram_prob = json.load(f)
    unigram_prob_tensor = torch.zeros(model_vocab_size)
    for word_id, prob in unigram_prob.items():
        unigram_prob_tensor[int(word_id)] = float(prob)

    unused_vocab_ids = set(range(model_vocab_size)) - set(unigram_prob.keys())
    return UnigramProb(prob=unigram_prob_tensor, unused_vocab_ids=unused_vocab_ids)


def load_unigram_prob_enwiki_vocab_min200(
    tokenizer: Union[
        sentence_transformers.models.tokenizer.WhitespaceTokenizer, WrappedTokenizer
    ],
    model_vocab_size: int,
    path: str = PATH_ENWIKI_VOCAB_MIN200,
    topk: Optional[int] = None,
) -> UnigramProb:
    """
    Load the unigram probabilities of the words in the vocabulary from the enwiki_vocab_min200.txt file.
    Only available for glove/word2vec. (Could be used for BERT-based models as well, but subword tokenization cause very sparse unigram probabilities without doing alignment)
    """
    frequency_dict: Dict[int, int] = {}
    # load the frequency of the words in the vocabulary
    with open(path, "r") as f:
        for count, line in enumerate(f):
            word_and_freq = line.rstrip().split(" ")
            assert (
                len(word_and_freq) == 2
            )  # ensuring that the line has only two elements, otherwise the file is not formatted correctly or the line is corrupted
            word, freq = word_and_freq
            freq = int(freq)
            word_id = tokenizer.word2idx[word] if word in tokenizer.word2idx else None
            if word_id is not None:
                frequency_dict[word_id] = freq
            if topk is not None and (count + 1) >= topk:
                break

    # create a tensor of the unigram probabilities
    unigram_prob = torch.zeros(model_vocab_size)
    for word_id, freq in frequency_dict.items():
        unigram_prob[word_id] = freq

    # normalize the unigram probabilities
    unigram_prob = unigram_prob / unigram_prob.sum()

    # check the top k most frequent words
    assert topk is None or len(frequency_dict) == topk

    unused_vocab_ids = set(range(model_vocab_size - 1)) - set(frequency_dict.keys())
    return UnigramProb(prob=unigram_prob, unused_vocab_ids=unused_vocab_ids)


def load_unigram_prob_jawiki_top_20k(
    tokenizer: Union[
        sentence_transformers.models.tokenizer.WhitespaceTokenizer, WrappedTokenizer
    ],
    model_vocab_size: int,
    path: str = PATH_JAWIKI_TOP_20K,
    topk: Optional[int] = None,
) -> UnigramProb:
    """
    Load the unigram probabilities of the words in the vocabulary from the enwiki_vocab_min200.txt file.
    """
    frequency_dict: Dict[int, int] = {}
    # load the frequency of the words in the vocabulary
    with open(path, "r") as f:
        for count, line in enumerate(f):
            word_and_freq = line.rstrip().split(" ")
            assert (
                len(word_and_freq) == 2
            )  # ensuring that the line has only two elements, otherwise the file is not formatted correctly or the line is corrupted
            word, freq = word_and_freq
            freq = int(freq)
            word_id = tokenizer.word2idx[word] if word in tokenizer.word2idx else None
            if word_id is not None:
                frequency_dict[word_id] = freq
            if topk is not None and (count + 1) >= topk:
                break

    # create a tensor of the unigram probabilities
    unigram_prob = torch.zeros(model_vocab_size)
    for word_id, freq in frequency_dict.items():
        unigram_prob[word_id] = freq

    # normalize the unigram probabilities
    unigram_prob = unigram_prob / unigram_prob.sum()

    # check the top k most frequent words
    assert topk is None or len(frequency_dict) == topk

    unused_vocab_ids = set(range(model_vocab_size - 1)) - set(frequency_dict.keys())
    return UnigramProb(prob=unigram_prob, unused_vocab_ids=unused_vocab_ids)


def load_unigram_prob_in_batch(
    tokenizer: Union[
        sentence_transformers.models.tokenizer.WhitespaceTokenizer, WrappedTokenizer
    ],
    model_vocab_size: int,
    task_name: str,
    split: str,
    topk: Optional[int] = None,
) -> UnigramProb:
    task_class = TASK_NAME_TO_DATASET[task_name]
    task = task_class()
    task.load_data()
    dataset = task.dataset[split]
    sentences = dataset["sentence1"] + dataset["sentence2"]
    frequency_dict: Dict[int, int] = {}
    print(f"Calculating the whole dataset freqeucny in {task_name}, {split}...")
    # calculate the frequency of the words in the dataset

    for sentence in tqdm(sentences):
        for word_id in tokenizer.tokenize(sentence):
            frequency_dict[word_id] = frequency_dict.get(word_id, 0) + 1
    print(
        f"Calculated frequency of {len(frequency_dict)} words, the sum of the frequency is {sum(frequency_dict.values())}"
    )

    # create a tensor of the unigram probabilities
    unigram_prob = torch.zeros(model_vocab_size)
    for word_id, freq in frequency_dict.items():
        unigram_prob[word_id] = freq

    # normalize the unigram probabilities
    unigram_prob = unigram_prob / unigram_prob.sum()

    # sort dict by frequency, descending
    frequency_dict = dict(
        sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    )
    # check the top k most frequent words

    unused_vocab_ids = set(range(model_vocab_size - 1)) - set(frequency_dict.keys())
    return UnigramProb(prob=unigram_prob, unused_vocab_ids=unused_vocab_ids)


def remove_unused_words(
    unused_vocab_ids: set[int],
    W: TT["num_words", "hidden_dim"],
    p: Optional[TT["num_words"]] = None,
) -> tuple[TT["num_words", "hidden_dim"], Optional[TT["num_words"]]]:
    """
    Remove the unused words from the input embeddings and unigram probabilities.
    If the topk is provided, only the topk most frequent words are kept.
    """
    W = W.clone()
    p = p.clone() if p is not None else None

    mask = torch.ones(W.shape[0], dtype=bool)
    mask[list(unused_vocab_ids)] = False
    W = W[mask]
    p = p[mask] if p is not None else None
    if p is None:
        return W, None
    else:
        p = p / p.sum()
        assert W.shape[0] == p.shape[0]
        return W, p


def load_word2vec_model(model_name: str, from_text_file=False) -> SentenceTransformer:
    embedding = (
        WordEmbeddings.from_text_file(model_name + ".txt")
        if from_text_file
        else WordEmbeddings.load(model_name)
    )
    pooling = Pooling(embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[embedding, pooling])
    return model


def load_fasttext_model(model_name: str) -> SentenceTransformer:
    # load the fasttext model from the text file
    embedding = WordEmbeddings.from_text_file(model_name)
    pooling = Pooling(embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[embedding, pooling])
    return model
