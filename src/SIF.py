from typing import Optional

import torch
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
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from torchtyping import TensorType as TT
from tqdm import tqdm
from typing import Tuple, Dict

from modeling import CustomPooling

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


class SIF:
    def __init__(
        self,
        model: SentenceTransformer,
        task_name: str,
        components_threshhold: int = 1,
        a: float = 10 ** (-3),
        data_split="test",  # test for STS-Benchmark, validation for word-level tasks
    ):
        pooling_layer_index = -1
        embedding_layer_index = 0

        self.components_threshhold = components_threshhold
        self.a = a
        self.is_fitted = False
        self.model = model

        task_class = TASK_NAME_TO_DATASET[task_name]
        task = task_class()
        task.load_data()
        self.dataset = task.dataset[data_split]
        self.sentences = self.dataset["sentence1"] + self.dataset["sentence2"]
        self.num_sentences = len(self.sentences)
        self.word_embedding_dimension = model[
            embedding_layer_index
        ].get_word_embedding_dimension()

    def calculate_sif_weights(self) -> TT["num_words"]:
        p = self.p
        return self.a / (self.a + p)

    def _embed_sentences(self) -> TT["num_sentences", "hidden_dim"]:
        # Since the evaluation in MTEB computes sentence embeddings one by one,
        # we need to pre-compute the embeddings for all sentences in the dataset here to get whole dataset embeddings.
        # This is not efficient, but it is the one of the simplest way to implement SIF to work with MTEB.
        # During the inference, the weighted average of the embeddings will be computed and then the sentence-level common component will be removed
        # inside the CustomPooling layer.
        encoded_sentences = self.model.encode(  # maybe too large batch size?
            self.sentences, convert_to_tensor=True, show_progress_bar=True
        )
        assert encoded_sentences.shape == (
            self.num_sentences,
            self.word_embedding_dimension,
        )

        return encoded_sentences

    def fit(self, W: TT["num_words", "hidden_dim"], p: Optional[TT["num_words"]]):
        """
        Args:
            W (TT["num_words", "hidden_dim"]): input embeddings. THIS IS NOT USED IN SIF SINCE IT IS FIT ON THE WHOLE DATASET.
            p (Optional[TT["num_words"]]): unigram probability.
            MAKE SURE that W and p is the "removed" version of the original W and p.
        """
        # hard coding for now. maybe won't work for bert-based models.
        pooling_layer_index = -1
        embedding_layer_index = 0

        self.p = p
        self.sif_weights: TT["num_words"] = self.calculate_sif_weights()

        # Pre-compute the embeddings for all sentences in the dataset for the component removal
        word_embedding_dimension = self.model[
            embedding_layer_index
        ].get_word_embedding_dimension()
        self.model[pooling_layer_index] = CustomPooling(
            word_embedding_dimension=self.word_embedding_dimension,
            weights=self.sif_weights,
            pooling_mode="sif_wo_component_removal",
        )
        S: TT["num_sentences", "hidden_dim"] = self._embed_sentences()
        S = S.cpu().numpy()
        # Apply sentence-level common component removal
        pca = PCA().fit(S)
        self.common_components: TT["D", "hidden_dim"] = pca.components_[
            : self.components_threshhold
        ]
        S = torch.tensor(S, device=self.model.device)
        self.common_components = torch.tensor(
            self.common_components, device=self.model.device
        )

        return self
