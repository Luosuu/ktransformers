'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import warnings
import torch.nn.functional as F
from ktransformers.operators.models import KLlamaModel
from ktransformers.models.configuration_deepseek import DeepseekV2Config
from ktransformers.models.configuration_llama import LlamaConfig
from ktransformers.models.modeling_llama import LlamaRotaryEmbedding
from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Attention
from ktransformers.models.modeling_deepseek_v3 import apply_rotary_pos_emb as apply_rotary_pos_emb_v3
from typing import Optional, Tuple
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from flash_attn import flash_attn_with_kvcache, flash_attn_func
from ktransformers.operators.triton_attention import decode_attention_fwd_grouped
logger = logging.getLogger("attention")


# V3 MLA is same to V2
class KDeepseekV2Attention(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].reshape(-1, self.kv_lora_rank)
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].reshape(-1, self.kv_lora_rank)
            self.q_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, 
                                      bias=False, dtype=q_absorb.dtype, device=q_absorb.device)
            self.q_absorb.weight.data = q_absorb
            self.out_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, 
                                        bias=False, dtype=out_absorb.dtype, device=out_absorb.device)
            self.out_absorb.weight.data = out_absorb
            #del self.orig_module.kv_b_proj
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return q_absorb, out_absorb

    def forward_chunck(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)
            #if cache_position is not None:  
            #    compressed_kv = compressed_kv[:,: cache_position[-1] + 1,:]
            #    k_pe = k_pe[:,:,: cache_position[-1] + 1,:]
        q_absorb, out_absorb = self.get_absorbed()

        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
        """
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        """
        if attention_mask is not None:
            """
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            """
            #causal_mask = attention_mask[:, :, :, : kv_seq_len]
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

        attn_output = torch.matmul(attn_output, out_absorb.mT) 

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        compressed_kv = compressed_kv.view(bsz, q_len, 1, self.kv_lora_rank)
        
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        k_pe = k_pe.transpose(1, 2) # [bsz, q_len, 1, self.qk_rope_head_dim]
  
        # decode
        if q_len == 1:
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                k_pe, compressed_kv, page_table = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
        
            # q_nope [bsz, self.num_heads, q_len, self.qk_nope_head_dim]
            # q_absorb [self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank]
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = torch.matmul(q_nope, q_absorb) # batched MM
            # q_nope [bsz, self.num_heads, q_len, self.kv_lora_rank]
            # q_pe [bsz, self.num_heads, q_len, self.qk_rope_head_dim]
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            # k_pe [bsz, q_len, 1, self.qk_rope_head_dim]
            # compressed_kv [bsz, q_len, 1, self.kv_lora_rank]
            key_states = torch.cat([compressed_kv, k_pe], dim=-1)
            
            query_states = query_states.squeeze(2)
            attn_output = torch.zeros_like(q_nope)
            
            attn_logits = torch.empty(
                    (
                        bsz,
                        self.num_heads,
                        1, #num_kv_splits # follow vLLM, fix it TODO
                        self.kv_lora_rank + 1, 
                    ),
                    dtype=torch.float32,
                    device = attn_output.device
                )

            """
            print("query_states", torch.isnan(query_states).any())
            print("key_states", torch.isnan(key_states[:,:,0,:]).any())
            print("compressed_kv", torch.isnan(compressed_kv[:,:,0,:]).any())
            print("position_ids", torch.isnan(position_ids).any())
            """

            # flash attn doesn't support head_dim bigger than 256
            # use vLLM triton attention kernel for MQA
            decode_attention_fwd_grouped(query_states, key_states, compressed_kv, attn_output,
                             page_table,
                             position_ids.squeeze(0).to(torch.int32), attn_logits,
                             1, #num_kv_splits # follow vLLM, fix it TODO
                             self.softmax_scale,
                             past_key_value.page_size)
            
            attn_output = torch.matmul(attn_output, out_absorb.mT) 
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output)
            
            #print("attn_output", torch.isnan(attn_output).any())
            return attn_output, None, past_key_value
        else:
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                k_pe.squeeze(0)
                compressed_kv.squeeze(0)
                past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
                k_pe.unsqueeze(0)
                compressed_kv.unsqueeze(0)
        
            k_pe = k_pe[:, :q_len]
            compressed_kv = compressed_kv[:, :q_len]
            kv = (
                self.kv_b_proj(compressed_kv)
                .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

            key_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
            
            query_states = query_states.transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_heads, self.v_head_dim)
            value_states_padded = torch.nn.functional.pad(value_states, [0, query_states.shape[-1] - value_states.shape[-1]], value=0)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states_padded,
                softmax_scale=self.softmax_scale,
                causal=True,
            )

            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, : self.v_head_dim]

            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.v_head_dim
            ).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class KLlamaAttention(BaseInjectedModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            logger.warning(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if q_len == 1:
            position_ids = position_ids[0][-1].unsqueeze(0).unsqueeze(0)
            query_states = query_states[:, :, -1:]
            key_states = key_states[:, :, -1:]

        attn_output = KLlamaModel.dynamic_sdpa.apply(
            self.layer_idx,
            bsz,
            position_ids[0][0],
            query_states.transpose(1, 2).to(torch.float16),
            key_states.transpose(1, 2).to(torch.float16),
            value_states.transpose(1, 2).to(torch.float16),
            mode="prefill" if q_len > 1 else "generate",
        )


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value