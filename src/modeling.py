import json
import os
from typing import Dict, Optional, Union

import torch
from IPython import embed
from torch import Tensor, nn
from torchtyping import TensorType as TT


def load_zipfian_whitening_norm(model_name: str) -> TT["num_words"]:
    """
    Helper function to load the zipfian whitening norm for the given model.
    """
    load_path = f"data/norm/zipfian/whitening/{model_name}.pt"
    return torch.load(load_path)


def load_uniform_centering_norm(model_name: str) -> TT["num_words"]:
    """
    Helper function to load the uniform centering norm for the given model.
    """
    load_path = f"data/norm/uniform/centering/{model_name}.pt"
    return torch.load(load_path)


def load_uniform_whitening_norm(model_name: str) -> TT["num_words"]:
    """
    Helper function to load the uniform whitening norm for the given model.
    """
    load_path = f"data/norm/uniform/whitening/{model_name}.pt"
    return torch.load(load_path)


# Custom pooling layer with Whitening.
# Originally from Sentence-Transformers: https://github.com/UKPLab/sentence-transformers/blob/90171f59893e423f1d1d1ff523fb241ff518e114/sentence_transformers/models/Pooling.py
class CustomPooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows
    to use the CLS token if it is returned by the underlying word embedding model. You can concatenate multiple poolings
    together.

    If pooling_mode_whitening AND centering, then only centering is performed (de-correlation is not performed).

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Either "cls", "lasttoken", "max", "mean", "mean_sqrt_len_tokens", or "weightedmean". If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling. See `SGPT: GPT Sentence Embeddings for Semantic Search <https://arxiv.org/abs/2202.08904>`_.
    :param pooling_mode_lasttoken: Perform last token pooling. See `SGPT: GPT Sentence Embeddings for Semantic Search <https://arxiv.org/abs/2202.08904>`_ and `Text and Code Embeddings by Contrastive Pre-Training <https://arxiv.org/abs/2201.10005>`_.
    :param pooling_mode_whitening: Perform mean-pooling after whitening the embeddings.
    """

    POOLING_MODES = (
        "cls",
        "lasttoken",
        "max",
        "mean",
        "mean_sqrt_len_tokens",
        "weightedmean",
        "whitening",
        "component_removal",
        "centering_only",
        "sif_wo_component_removal",
        "sif_w_component_removal",
        "uniform_centering_then_zipfian_whitening_norm",
        "uniform_whitening_then_zipfian_whitening_norm",
        "zipfian_whitening_then_uniform_centering_norm",
        "zipfian_whitening_then_uniform_whitening_norm",
        "raw_then_zipfian_whitening_norm",  # TODO: naming inconsistency
        "raw_then_zipfian_whitening_dirction",  # TODO: naming inconsistency
    )

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = False,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
        pooling_mode_whitening: bool = True,
        pooling_mode_component_removal: bool = False,
        pooling_mode_centering_only: bool = False,
        pooling_mode_sif_wo_component_removal: bool = False,
        pooling_mode_sif_w_component_removal: bool = False,
        pooling_mode_uniform_centering_then_zipfian_whitening_norm: bool = False,
        pooling_mode_uniform_whitening_then_zipfian_whitening_norm: bool = False,
        pooling_mode_zipfian_whitening_then_uniform_centering_norm: bool = False,
        pooling_mode_zipfian_whitening_then_uniform_whitening_norm: bool = False,
        pooling_mode_raw_then_zipfian_whitening_norm: bool = False,
        pooling_mode_raw_then_zipfian_whitening_dirction: bool = False,
        include_prompt=True,
        whitening_transformer=None,
        weights=None,
        model_name: Optional[str] = None,
    ) -> None:
        super(CustomPooling, self).__init__()
        self.whitening_transformer = whitening_transformer
        self.weights = weights  # weights for the weighted mean pooling (SIF)
        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
            "pooling_mode_whitening",
            "pooling_mode_component_removal",
            "pooling_mode_centering_only",
            "pooling_mode_sif_wo_component_removal",
            "pooling_mode_sif_w_component_removal",
            "pooling_mode_uniform_centering_then_zipfian_whitening_norm",
            "pooling_mode_uniform_whitening_then_zipfian_whitening_norm",
            "pooling_mode_zipfian_whitening_then_uniform_centering_norm",
            "pooling_mode_zipfian_whitening_then_uniform_whitening_norm",
            "pooling_mode_raw_then_zipfian_whitening_norm",
            "pooling_mode_raw_then_zipfian_whitening_dirction",
        ]

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()

            if pooling_mode not in self.POOLING_MODES:
                raise ValueError(
                    f"Set invalid pooling mode: {pooling_mode}. Valid pooling modes are: {self.POOLING_MODES}."
                )

            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_mean_sqrt_len_tokens = pooling_mode == "mean_sqrt_len_tokens"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"
            pooling_mode_whitening = pooling_mode == "whitening"
            pooling_mode_component_removal = pooling_mode == "component_removal"
            pooling_mode_centering_only = pooling_mode == "centering_only"
            pooling_mode_sif_wo_component_removal = (
                pooling_mode == "sif_wo_component_removal"
            )
            pooling_mode_sif_w_component_removal = (
                pooling_mode == "sif_w_component_removal"
            )
            pooling_mode_uniform_centering_then_zipfian_whitening_norm = (
                pooling_mode == "uniform_centering_then_zipfian_whitening_norm"
            )
            pooling_mode_uniform_whitening_then_zipfian_whitening_norm = (
                pooling_mode == "uniform_whitening_then_zipfian_whitening_norm"
            )
            pooling_mode_zipfian_whitening_then_uniform_centering_norm = (
                pooling_mode == "zipfian_whitening_then_uniform_centering_norm"
            )
            pooling_mode_zipfian_whitening_then_uniform_whitening_norm = (
                pooling_mode == "zipfian_whitening_then_uniform_whitening_norm"
            )
            pooling_mode_raw_then_zipfian_whitening_norm = (
                pooling_mode == "raw_then_zipfian_whitening_norm"
            )
            pooling_mode_raw_then_zipfian_whitening_dirction = (
                pooling_mode == "raw_then_zipfian_whitening_dirction"
            )

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken
        self.pooling_mode_whitening = pooling_mode_whitening
        self.pooling_mode_component_removal = pooling_mode_component_removal
        self.pooling_mode_centering_only = pooling_mode_centering_only
        self.pooling_mode_sif_wo_component_removal = (
            pooling_mode_sif_wo_component_removal
        )
        self.pooling_mode_sif_w_component_removal = pooling_mode_sif_w_component_removal
        self.pooling_mode_uniform_centering_then_zipfian_whitening_norm = (
            pooling_mode_uniform_centering_then_zipfian_whitening_norm
        )
        self.pooling_mode_uniform_whitening_then_zipfian_whitening_norm = (
            pooling_mode_uniform_whitening_then_zipfian_whitening_norm
        )
        self.pooling_mode_zipfian_whitening_then_uniform_centering_norm = (
            pooling_mode_zipfian_whitening_then_uniform_centering_norm
        )
        self.pooling_mode_zipfian_whitening_then_uniform_whitening_norm = (
            pooling_mode_zipfian_whitening_then_uniform_whitening_norm
        )
        self.pooling_mode_raw_then_zipfian_whitening_norm = (
            pooling_mode_raw_then_zipfian_whitening_norm
        )
        self.pooling_mode_raw_then_zipfian_whitening_dirction = (
            pooling_mode_raw_then_zipfian_whitening_dirction
        )

        self.include_prompt = include_prompt

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
                pooling_mode_whitening,
                pooling_mode_component_removal,
                pooling_mode_centering_only,
                pooling_mode_sif_wo_component_removal,
                pooling_mode_sif_w_component_removal,
                pooling_mode_uniform_centering_then_zipfian_whitening_norm,
                pooling_mode_uniform_whitening_then_zipfian_whitening_norm,
                pooling_mode_zipfian_whitening_then_uniform_centering_norm,
                pooling_mode_zipfian_whitening_then_uniform_whitening_norm,
                pooling_mode_raw_then_zipfian_whitening_norm,
                pooling_mode_raw_then_zipfian_whitening_dirction,
            ]
        )
        self.pooling_output_dimension = (
            pooling_mode_multiplier * word_embedding_dimension
        )

        self.model_name = model_name

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_weightedmean_tokens:
            modes.append("weightedmean")
        if self.pooling_mode_lasttoken:
            modes.append("lasttoken")
        if self.pooling_mode_whitening:
            modes.append("whitening")
        if self.pooling_mode_component_removal:
            modes.append("component_removal")
        if self.pooling_mode_centering_only:
            modes.append("centering_only")
        if self.pooling_mode_sif_wo_component_removal:
            modes.append("sif_wo_component_removal")
        if self.pooling_mode_sif_w_component_removal:
            modes.append("sif_w_component_removal")
        if self.pooling_mode_uniform_centering_then_zipfian_whitening_norm:
            modes.append("uniform_centering_then_zipfian_whitening_norm")
        if self.pooling_mode_uniform_whitening_then_zipfian_whitening_norm:
            modes.append("uniform_whitening_then_zipfian_whitening_norm")
        if self.pooling_mode_zipfian_whitening_then_uniform_centering_norm:
            modes.append("zipfian_whitening_then_uniform_centering_norm")
        if self.pooling_mode_zipfian_whitening_then_uniform_whitening_norm:
            modes.append("zipfian_whitening_then_uniform_whitening_norm")
        if self.pooling_mode_raw_then_zipfian_whitening_norm:
            modes.append("raw_then_zipfian_whitening_norm")
        if self.pooling_mode_raw_then_zipfian_whitening_dirction:
            modes.append("raw_then_zipfian_whitening_dirction")

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]
        if not self.include_prompt and "prompt_length" in features:
            attention_mask[:, : features["prompt_length"]] = 0
        input_ids = features["input_ids"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get(
                "cls_token_embeddings", token_embeddings[:, 0]
            )  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = (
                    features["token_weights_sum"]
                    .unsqueeze(-1)
                    .expand(sum_embeddings.size())
                )
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = (
                    features["token_weights_sum"]
                    .unsqueeze(-1)
                    .expand(sum_embeddings.size())
                )
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        ####################################
        # Custom pooling for centering only#
        ####################################
        if self.pooling_mode_centering_only:
            # perform centering only (de-correlation nor common-component removal is not performed)
            token_embeddings -= self.whitening_transformer.mu
            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        ###############################
        # Custom pooling for whitening#
        ###############################
        if self.pooling_mode_whitening:
            token_embeddings -= self.whitening_transformer.mu
            # pefrom whitening
            token_embeddings = (
                token_embeddings @ self.whitening_transformer.transformation_matrix
            )
            # peform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        #########################################################
        # Custom pooling for component removal (mainly for abtp)#
        #########################################################
        if self.pooling_mode_component_removal:
            # perform centering
            token_embeddings -= self.whitening_transformer.mu
            # perform component removal
            token_embeddings = token_embeddings - (
                (token_embeddings @ self.whitening_transformer.common_components.T)
                @ self.whitening_transformer.common_components
            )
            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        #########################################################
        # Custom pooling for component SIF wo component removal##
        # Mainly for pre-embedding coputation for SIF ###########
        # before the common component removal in the SIF layer###
        # Only used for the training#############################
        #########################################################
        if self.pooling_mode_sif_wo_component_removal:
            sif_weights: TT["num_words"] = self.weights
            # look-up the sif weights for input_ids
            # Inputs : sif_weights: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: sif_weights: TT["batch_size", "num_words", "hidden_dim"]
            sif_weights_expanded: TT["batch_size", "num_vocab"] = sif_weights.unsqueeze(
                0
            ).expand(input_ids.size(0), -1)
            sif_weights_expanded: TT["batrch_size", "nun_words"] = torch.gather(
                sif_weights_expanded, 1, input_ids
            )  # look-up the sif weights for input_ids
            sif_weights_expanded: TT["batch_size", "num_words", "hidden_dim"] = (
                sif_weights_expanded.unsqueeze(
                    -1
                ).expand(-1, -1, token_embeddings.size(-1))
            )
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded * sif_weights_expanded, 1
            )
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        #########################################################
        # Custom pooling for component SIF w component removal###
        # Mainly for the inference after the common component ###
        # removal in the SIF layer###############################
        #########################################################
        if self.pooling_mode_sif_w_component_removal:
            sif_weights: TT["num_words"] = self.whitening_transformer.sif_weights
            # look-up the sif weights for input_ids
            # Inputs : sif_weights: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: sif_weights: TT["batch_size", "num_words", "hidden_dim"]
            sif_weights_expanded: TT["batch_size", "num_vocab"] = sif_weights.unsqueeze(
                0
            ).expand(input_ids.size(0), -1)
            sif_weights_expanded: TT["batrch_size", "nun_words"] = torch.gather(
                sif_weights_expanded, 1, input_ids
            )  # look-up the sif weights for input_ids
            sif_weights_expanded: TT["batch_size", "num_words", "hidden_dim"] = (
                sif_weights_expanded.unsqueeze(
                    -1
                ).expand(-1, -1, token_embeddings.size(-1))
            )
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded * sif_weights_expanded, 1
            )
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sum_embeddings = sum_embeddings / sum_mask
            # Sentence-level common component removal
            sum_embeddings = sum_embeddings - (
                (sum_embeddings @ self.whitening_transformer.common_components.T)
                @ self.whitening_transformer.common_components
            )
            output_vectors.append(sum_embeddings)
        #####################################################################
        # Custom pooling for uniform centering then zipfian whitening norm###
        #####################################################################
        if self.pooling_mode_uniform_centering_then_zipfian_whitening_norm:
            zipfian_whitening_norm: TT["num_words"] = load_zipfian_whitening_norm(
                self.model_name
            )
            # look up the zipfian whitening norm for input_ids
            # Inputs : zipfian_whitening_norm: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: zipfian_whitening_norm: TT["batch_size", "num_words", "hidden_dim"]
            zipfian_whitening_norm_expanded: TT["batch_size", "num_vocab"] = (
                zipfian_whitening_norm.unsqueeze(0).expand(input_ids.size(0), -1)
            )
            zipfian_whitening_norm_expanded: TT["batrch_size", "nun_words"] = (
                torch.gather(zipfian_whitening_norm_expanded, 1, input_ids)
            )  # look-up the zipfian whitening norm for input_ids
            zipfian_whitening_norm_expanded: TT[
                "batch_size", "num_words", "hidden_dim"
            ] = zipfian_whitening_norm_expanded.unsqueeze(-1).expand(
                -1, -1, token_embeddings.size(-1)
            )

            # perform centering
            token_embeddings -= self.whitening_transformer.mu

            # normalize the norm of the embeddings to one
            token_embeddings = token_embeddings / torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * zipfian_whitening_norm_expanded,  # multiply the embeddings by the zipfian whitening norm
                1,
            )
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        #####################################################################
        # Custom pooling for uniform whitening then zipfian whitening norm###
        #####################################################################
        if self.pooling_mode_uniform_whitening_then_zipfian_whitening_norm:
            zipfian_whitening_norm: TT["num_words"] = load_zipfian_whitening_norm(
                self.model_name
            )
            # look up the zipfian whitening norm for input_ids
            # Inputs : zipfian_whitening_norm: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: zipfian_whitening_norm: TT["batch_size", "num_words", "hidden_dim"]
            zipfian_whitening_norm_expanded: TT["batch_size", "num_vocab"] = (
                zipfian_whitening_norm.unsqueeze(0).expand(input_ids.size(0), -1)
            )
            zipfian_whitening_norm_expanded: TT["batrch_size", "nun_words"] = (
                torch.gather(zipfian_whitening_norm_expanded, 1, input_ids)
            )  # look-up the zipfian whitening norm for input_ids
            zipfian_whitening_norm_expanded: TT[
                "batch_size", "num_words", "hidden_dim"
            ] = zipfian_whitening_norm_expanded.unsqueeze(-1).expand(
                -1, -1, token_embeddings.size(-1)
            )

            # perform centering
            token_embeddings -= self.whitening_transformer.mu

            # perform whitening
            token_embeddings = (
                token_embeddings @ self.whitening_transformer.transformation_matrix
            )

            # normalize the norm of the embeddings to one
            token_embeddings = token_embeddings / torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * zipfian_whitening_norm_expanded,  # multiply the embeddings by the zipfian whitening norm
                1,
            )
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        #####################################################################
        # Custom pooling for zipfian whitening then uniform centering norm###
        #####################################################################
        if self.pooling_mode_zipfian_whitening_then_uniform_centering_norm:
            uniform_centering_norm: TT["num_words"] = load_uniform_centering_norm(
                self.model_name
            )
            # look up the uniform centering norm for input_ids
            # Inputs : uniform_centering_norm: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: uniform_centering_norm: TT["batch_size", "num_words", "hidden_dim"]
            uniform_centering_norm_expanded: TT["batch_size", "num_vocab"] = (
                uniform_centering_norm.unsqueeze(0).expand(input_ids.size(0), -1)
            )
            uniform_centering_norm_expanded: TT["batrch_size", "nun_words"] = (
                torch.gather(uniform_centering_norm_expanded, 1, input_ids)
            )  # look-up the uniform centering norm for input_ids
            uniform_centering_norm_expanded: TT[
                "batch_size", "num_words", "hidden_dim"
            ] = uniform_centering_norm_expanded.unsqueeze(-1).expand(
                -1, -1, token_embeddings.size(-1)
            )

            # perform centering
            token_embeddings -= self.whitening_transformer.mu

            # perform whitening
            token_embeddings = (
                token_embeddings @ self.whitening_transformer.transformation_matrix
            )

            # normalize the norm of the embeddings to one
            token_embeddings = token_embeddings / torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )

            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * uniform_centering_norm_expanded,  # multiply the embeddings by the uniform centering norm
                1,
            )

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        #####################################################################
        # Custom pooling for zipfian whitening then uniform whitening norm###
        #####################################################################
        if self.pooling_mode_zipfian_whitening_then_uniform_whitening_norm:
            uniform_whitening_norm: TT["num_words"] = load_uniform_whitening_norm(
                self.model_name
            )
            # look up the uniform whitening norm for input_ids
            # Inputs : uniform_whitening_norm: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: uniform_whitening_norm: TT["batch_size", "num_words", "hidden_dim"]
            uniform_whitening_norm_expanded: TT["batch_size", "num_vocab"] = (
                uniform_whitening_norm.unsqueeze(0).expand(input_ids.size(0), -1)
            )
            uniform_whitening_norm_expanded: TT["batrch_size", "nun_words"] = (
                torch.gather(uniform_whitening_norm_expanded, 1, input_ids)
            )
            uniform_whitening_norm_expanded: TT[
                "batch_size", "num_words", "hidden_dim"
            ] = uniform_whitening_norm_expanded.unsqueeze(-1).expand(
                -1, -1, token_embeddings.size(-1)
            )

            # perform centering
            token_embeddings -= self.whitening_transformer.mu

            # perform whitening
            token_embeddings = (
                token_embeddings @ self.whitening_transformer.transformation_matrix
            )

            # normalize the norm of the embeddings to one
            token_embeddings = token_embeddings / torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )

            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * uniform_whitening_norm_expanded,  # multiply the embeddings by the uniform whitening norm
                1,
            )

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        #####################################################################
        # Custom pooling for raw then zipfian whitening norm#################
        #####################################################################
        if self.pooling_mode_raw_then_zipfian_whitening_norm:
            # get zipfian whitening norm
            zipfian_whitening_norm: TT["num_words"] = load_zipfian_whitening_norm(
                self.model_name
            )
            # look up the zipfian whitening norm for input_ids
            # Inputs : zipfian_whitening_norm: TT["num_words"]
            #          input_ids: TT["batch_size", "num_words"]
            # Outputs: zipfian_whitening_norm: TT["batch_size", "num_words", "hidden_dim"]
            zipfian_whitening_norm_expanded: TT["batch_size", "num_vocab"] = (
                zipfian_whitening_norm.unsqueeze(0).expand(input_ids.size(0), -1)
            )
            zipfian_whitening_norm_expanded: TT["batrch_size", "nun_words"] = (
                torch.gather(zipfian_whitening_norm_expanded, 1, input_ids)
            )
            zipfian_whitening_norm_expanded: TT[
                "batch_size", "num_words", "hidden_dim"
            ] = zipfian_whitening_norm_expanded.unsqueeze(-1).expand(
                -1, -1, token_embeddings.size(-1)
            )

            # do not apply centering nor whitening, use raw embeddings as is

            token_embeddings_norm = torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )
            # Ensure no zero norms by clamping to small positive value
            token_embeddings_norm = torch.clamp(token_embeddings_norm, min=1e-12)
            token_embeddings = token_embeddings / token_embeddings_norm

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )

            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * zipfian_whitening_norm_expanded,  # multiply the embeddings by the zipfian whitening norm
                1,
            )

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        #####################################################################
        # Custom pooling for raw then zipfian whitening direction############
        #####################################################################
        if self.pooling_mode_raw_then_zipfian_whitening_dirction:
            # step:
            # 1. copy the raw embeddings norm so that it can be used as multipliers later (bs, num_words)
            # 2. apply zipfian whitening to the embeddings
            # 3. normalize the zipfian whitened embeddings
            # 4. multiply the normalized embeddings by the copied raw embeddings norm

            # get the raw embeddings norm
            raw_embeddings_norm = torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )  # (bs, num_words, 1)

            # apply centering
            token_embeddings -= self.whitening_transformer.mu

            # apply whitening
            token_embeddings = (
                token_embeddings @ self.whitening_transformer.transformation_matrix
            )

            # normalize the norm of the embeddings to one
            token_embeddings = token_embeddings / torch.linalg.norm(
                token_embeddings, dim=-1, keepdim=True
            )

            # multiply the normalized embeddings by the raw embeddings norm

            # perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )

            sum_embeddings = torch.sum(
                token_embeddings
                * input_mask_expanded
                * raw_embeddings_norm,  # multiply the embeddings by the raw embeddings norm
                1,
            )

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # Use flip and max() to get the last index of 1 in the attention mask

            if torch.jit.is_tracing():
                # Avoid tracing the argmax with int64 input that can not be handled by ONNX Runtime: https://github.com/microsoft/onnxruntime/issues/10068
                attention_mask = attention_mask.to(torch.int32)

            values, indices = attention_mask.flip(1).max(1)
            indices = torch.where(values == 0, seq_len - 1, indices)
            gather_indices = seq_len - indices - 1

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
            )
            embedding = torch.gather(
                token_embeddings * input_mask_expanded, 1, gather_indices
            ).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return CustomPooling(**config)
