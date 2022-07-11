# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pickle

import numpy as np
import torch
from tqdm import trange

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
)
from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.input_features.qa_input_features import GPTQAInputFeatures
from nemo.utils import logging


class GPTQADataset(QADataset):
    """ Creates a Dataset for GPT architecture based Generative QA """

    def __init__(
        self,
        data_file: str,
        processor: object,
        tokenizer: object,
        keep_doc_spans: str = False,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 512,
        max_answer_length: int = 64,
        num_samples: int = -1,
        mode: str = TRAINING_MODE,
        use_cache: bool = False,
    ):
        super().__init__(data_file=data_file, processor=processor, tokenizer=tokenizer)

        self.keep_doc_spans = keep_doc_spans
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.num_samples = num_samples
        self.mode = mode
        self.use_cache = use_cache

        self._check_valid_mode()

        # get examples from processor and keep according to limit
        self.examples = self.processor.get_examples()
        if num_samples == 0:
            raise ValueError(
                f"num_samples has to be positive or -1 (to use the entire dataset), however got {num_samples}."
            )
        elif num_samples > 0:
            self.examples = self.examples[:num_samples]

        # create cached filename to load/save features as per flag
        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        self.cached_features_file = (
            self.data_file
            + '_cache'
            + '_{}_{}_{}_{}_{}_{}_{}'.format(
                self.mode,
                self.tokenizer.name,
                str(vocab_size),
                str(self.max_query_length),
                str(self.max_seq_length),
                str(self.max_answer_length),
                str(self.num_samples),
            )
        )

        if use_cache and os.path.exists(self.cached_features_file):
            self._load_features_from_cache()
        else:
            self._convert_examples_to_features()
            if use_cache:
                self._dump_features_to_cache()

        logging.info("Converting dict features into object features")
        for i in trange(len(self.features)):
            self.features[i] = GPTQAInputFeatures(**self.features[i])

    def _convert_examples_to_features(self):
        logging.info(f"Preprocessing data into features.")

        has_groundtruth = self.mode != INFERENCE_MODE
        unique_id = 1000000000
        self.features = []

        context_prefix = "context: "
        answer_prefix = " answer:"
        query_prefix = " question: "

        context_prefix_tokens = self.tokenizer.tokenizer.tokenize(context_prefix)
        answer_prefix_tokens = self.tokenizer.tokenizer.tokenize(answer_prefix)

        for example_index in trange(len(self.examples)):
            if example_index % 1000 == 0:
                GPTQADataset.check_if_sufficient_memory()

            example = self.examples[example_index]

            query_tokens_length, query_tokens, formatted_query = self._prep_query(query_prefix, example)
            answer_tokens_length, answer_tokens, formatted_answer = self._prep_answer(example, has_groundtruth)
            context_tokens, context_spans = self._prep_context(
                example, query_tokens_length, answer_tokens_length, context_prefix_tokens, answer_prefix_tokens,
            )

            unique_id = self._encode_all_context_spans(
                unique_id,
                context_spans,
                context_tokens,
                context_prefix,
                formatted_query,
                answer_prefix,
                formatted_answer,
                example,
                example_index,
            )

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            self.examples = []
            del self.processor

    def _prep_query(self, query_prefix, example):
        formatted_query = f"{query_prefix}{example.question_text}"
        query_tokens_length, query_tokens, formatted_query = self._get_n_tokens_in_sentence(
            formatted_query, self.max_query_length
        )

        return query_tokens_length, query_tokens, formatted_query

    def _prep_answer(self, example, has_groundtruth):
        if has_groundtruth:
            if not example.is_impossible:
                target = f"{example.answer_text}{self.tokenizer.tokenizer.eos_token}"
            else:
                target = self.tokenizer.tokenizer.eos_token
        else:
            target = self.tokenizer.tokenizer.eos_token

        if target:
            answer_tokens_length, answer_tokens, formatted_answer = self._get_n_tokens_in_sentence(
                target, self.max_answer_length
            )
        else:
            answer_tokens_length, answer_tokens, formatted_answer = 0, None, None

        return answer_tokens_length, answer_tokens, formatted_answer

    def _prep_context(
        self, example, query_tokens_length, answer_tokens_length, context_prefix_tokens, answer_prefix_tokens,
    ):
        context_tokens = self.tokenizer.tokenizer.tokenize(example.context_text)

        # -1 accounts for EOS token
        max_context_length = (
            self.max_seq_length
            - query_tokens_length
            - answer_tokens_length
            - len(context_prefix_tokens)
            - len(answer_prefix_tokens)
            - 1
        )
        context_spans = GPTQADataset.get_docspans(context_tokens, max_context_length, self.doc_stride)
        context_spans = tuple(context_spans)

        return context_tokens, context_spans

    def _encode_all_context_spans(
        self,
        unique_id,
        context_spans,
        context_tokens,
        context_prefix,
        formatted_query,
        answer_prefix,
        formatted_answer,
        example,
        example_index,
    ):
        for context_span_idx, context_span in enumerate(context_spans):
            context_span_tokens = context_tokens[context_span.start : context_span.start + context_span.length]
            context_span_text = self.tokenizer.tokenizer.convert_tokens_to_string(context_span_tokens)

            input_without_answer = f"{context_prefix}{context_span_text}{formatted_query}{answer_prefix}"
            training_mask_end, _, _ = self._get_n_tokens_in_sentence(input_without_answer, self.max_seq_length)

            # mark as a negative sample if answer not present in context
            if formatted_answer and example.answer_text:
                if example.answer_text in context_span_text:
                    input_with_answer = f"{input_without_answer}{formatted_answer}"
                else:
                    input_with_answer = f"{input_without_answer}{self.tokenizer.tokenizer.eos_token}"
            else:
                input_with_answer = f"{input_without_answer}{self.tokenizer.tokenizer.eos_token}"

            encoded_input_dict = self.tokenizer.tokenizer(
                input_with_answer,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = torch.squeeze(encoded_input_dict["input_ids"])
            input_attn_mask = torch.squeeze(encoded_input_dict["attention_mask"])

            labels = GPTQADataset.get_labels_from_inputs(
                input_ids, training_mask_end, input_attn_mask, self.tokenizer.tokenizer,
            )

            # create dictionary features
            feature = {
                "unique_id": unique_id,
                "input_ids": input_ids,
                "input_attn_mask": input_attn_mask,
                "training_mask_end": training_mask_end,
                "labels": labels,
                "example_index": example_index,
                "context_span_index": context_span_idx,
                "is_impossible": example.is_impossible,
            }

            self.features.append(feature)
            unique_id += 1

        return unique_id

    def _get_n_tokens_in_sentence(self, sentence, max_length):
        tokens = self.tokenizer.tokenizer.tokenize(sentence)[:max_length]
        trunc_sentence = self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
        seq_length = len(tokens)

        return seq_length, tokens, trunc_sentence

    def _load_features_from_cache(self):
        """ Loads pickled features from the file """

        logging.info(f"loading from {self.cached_features_file}")

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            del self.examples
            del self.processor

        with open(self.cached_features_file, "rb") as reader:
            self.features = pickle.load(reader)

    def _dump_features_to_cache(self):
        """ Pickles features to file """

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if master_device:
            logging.info(f"Saving train features into cached file {self.cached_features_file}")
            with open(self.cached_features_file, "wb") as writer:
                pickle.dump(self.features, writer)

    def _check_valid_mode(self):
        """ Checks if provided mode is in given three options """

        if self.mode not in [TRAINING_MODE, EVALUATION_MODE, INFERENCE_MODE]:
            raise ValueError(
                f"mode should be either {TRAINING_MODE}, {EVALUATION_MODE}, {INFERENCE_MODE} but got {self.mode}"
            )
        else:
            return

    @classmethod
    def get_labels_from_inputs(cls, input_ids, training_mask_end, input_attn_mask, tokenizer):
        """
        Loss mask for GPT is constructed to ignore loss for padding tokens
        GPT eos token is same as pas token and needs to be excluded from loss mask
        This is done using the attention mask inversion as described in:
            https://github.com/huggingface/transformers/issues/7135#issuecomment-1172962080
        """
        labels = copy.copy(torch.squeeze(input_ids))
        inv_bool_attn_mask = torch.eq(torch.squeeze(input_attn_mask), 0)
        labels.data = torch.tensor(
            [
                -100 if ((i < training_mask_end) or (inv_bool_attn_mask[i])) else labels.data[i]
                for i in range(len(labels.data))
            ]
        )

        return labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        if self.mode == INFERENCE_MODE:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.training_mask_end),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.training_mask_end),
                np.array(feature.labels),
            )