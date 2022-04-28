# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import MegatronTokenLevelHead
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    init_method_normal,
    scaled_init_method_normal,
)

try:
    from apex.transformer import tensor_parallel
    from apex.transformer.enums import ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()


__all__ = ["MegatronRetrievalTokenLevelEncoderDecoderModule"]


class MegatronRetrievalTokenLevelEncoderDecoderModule(MegatronModule):
    """Token-based (input/output is tokens) retrieval encoder-decoder model"""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_cross_entropy=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        layernorm_epsilon=1e-5,
        persist_layer_norm=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        openai_gelu=False,
        activation='gelu',
        onnx_safe=False,
        bias=True,
        hidden_steps=-1,
        hidden_blocks=1,
        add_encoder=True,
        add_decoder=True,
        chunk_size=64,
        enc_num_layers=4,  # total number of encoder layers
        dec_num_layers=6,  # total number of decoder layers
        enc_cross_attention=[3],  # layer numbers for cross attention
        dec_cross_attention=[3, 5],  # layer numbers for chunked cross attention
        add_position_embedding=False,
    ):
        super(MegatronRetrievalTokenLevelEncoderDecoderModule, self).__init__()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_cross_entropy = fp16_cross_entropy
        self.precision = precision
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        if pre_process:
            self.encoder_embedding = Embedding(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                max_sequence_length=max_position_embeddings,
                init_method=init_method_normal(init_method_std),
                num_tokentypes=num_tokentypes,
                use_cpu_initialization=use_cpu_initialization,
                embedding_dropout_prob=hidden_dropout,
                add_position_embedding=add_position_embedding,
            )
            self._embedding_key = "embedding"

        if add_encoder:
            enc_layer_types = []
            for i in range(enc_num_layers):
                if i in enc_cross_attention:
                    enc_layer_types.append(LayerType.retrieval_encoder)
                else:
                    enc_layer_types.append(LayerType.encoder)
            self.encoder = get_encoder_model(
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=enc_num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=init_method_normal(init_method_std),
                scaled_init_method=scaled_init_method_normal(init_method_std, enc_num_layers),
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=init_method_std,
                use_cpu_initialization=use_cpu_initialization,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                layernorm_epsilon=layernorm_epsilon,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                hidden_blocks=hidden_blocks,
                activation=activation,
                bias=bias,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=enc_layer_types,
                chunk_size=chunk_size,
            )
            self._encoder_key = "encoder"

        if add_decoder:
            pre_decoder_num_layers = min(dec_cross_attention)
            pre_decoder_layer_types = []
            for i in range(pre_decoder_num_layers):
                pre_decoder_layer_types.append(LayerType.encoder)

            post_decoder_num_layers = dec_num_layers - pre_decoder_num_layers
            post_decoder_layer_types = []
            for i in range(post_decoder_num_layers):
                if i + pre_decoder_num_layers in dec_cross_attention:
                    post_decoder_layer_types.append(LayerType.retrieval_decoder)
                else:
                    post_decoder_layer_types.append(LayerType.encoder)

            # it is used to process the inputs for encoder to use as context (H in the paper)
            self.pre_decoder = get_decoder_model(
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=pre_decoder_num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=init_method_normal(init_method_std),
                scaled_init_method=scaled_init_method_normal(init_method_std, pre_decoder_num_layers),
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=init_method_std,
                use_cpu_initialization=use_cpu_initialization,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                layernorm_epsilon=layernorm_epsilon,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                hidden_blocks=hidden_blocks,
                activation=activation,
                bias=bias,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=pre_decoder_layer_types,
                chunk_size=chunk_size,
            )

            # it is where the chunked cross attention happens
            self.post_decoder = get_decoder_model(
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=post_decoder_num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=init_method_normal(init_method_std),
                scaled_init_method=scaled_init_method_normal(init_method_std, post_decoder_num_layers),
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=init_method_std,
                use_cpu_initialization=use_cpu_initialization,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                layernorm_epsilon=layernorm_epsilon,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                hidden_blocks=hidden_blocks,
                activation=activation,
                bias=bias,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=post_decoder_layer_types,
                chunk_size=chunk_size,
            )
            self._pre_decoder_key = "pre_decoder"
            self._post_decoder_key = "post_decoder"

        self.initialize_word_embeddings(
            init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
        )

        if add_decoder and post_process:
            self.tokens_head = MegatronTokenLevelHead(self.word_embeddings_weight().size(0), parallel_output)
            self._tokens_head_key = 'tokens_head'

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_emb,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        if input_emb is None:
            if self.pre_process and self.add_encoder:
                # encoder embeddings
                input_position_ids = build_position_ids(input_ids)
                input_emb = self.encoder_embedding(input_ids, input_position_ids, token_type_ids=token_type_ids)
            else:
                input_emb = None

        if self.add_decoder:
            hidden = self.pre_decoder(input_emb, input_attn_mask)

        if self.add_encoder:
            retrieved_emb = self.encoder(
                retrieved_emb, retrieved_attn_mask, context_attn_mask=input_attn_mask, encoder_output=hidden
            )

        if self.add_decoder:
            dec_output = self.post_decoder(
                hidden, input_attn_mask, context_attn_mask=retrieved_attn_mask, encoder_output=retrieved_emb
            )
            token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())

            if labels is not None:
                # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                if self.fp16_cross_entropy:
                    assert token_logits.dtype == torch.half
                    tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits, labels)
                else:
                    tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits.float(), labels)
                return tokens_loss
            else:
                return token_logits

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}

        state_dict_[self._encoder_embedding_key] = self.encoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._pre_decoder_key] = self.pre_decoder.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._post_decoder_key] = self.post_decoder.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._tokens_head_key] = self.tokens_head.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder_embedding.encoder_embeddingload_state_dict(state_dict[self._encoder_embedding_key], strict=strict)
        self.encoder.load_state_dict(state_dict[self._encoder_key], strict=strict)
        self.pre_decoder.load_state_dict(state_dict[self._pre_decoder_key], strict=strict)
        self.post_decoder.load_state_dict(state_dict[self._post_decoder_key], strict=strict)
        self.tokens_head.load_state_dict(state_dict[self._tokens_head_key], strict=strict)