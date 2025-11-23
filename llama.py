from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *

# Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
# borrowed from the official Llama implementation:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        按 RMSNorm 论文公式做归一化：
        - 对最后一维求均方值 mean(x^2)
        - 加上 eps，防止除以 0
        - 开平方得到 RMS，然后用 x / RMS 做归一化
        """
        # 对最后一维求均方值，保持维度
        rms_sq = x.pow(2).mean(dim=-1, keepdim=True)
        # 加 eps 再开方，避免数值不稳定
        denom = torch.sqrt(rms_sq + self.eps)
        # 用 RMS 做归一化
        return x / denom



    def forward(self, x):
        """
        Apply the root mean square normalizer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor) -> torch.Tensor:
        '''
        实现多头缩放点积注意力（Scaled Dot-Product Attention）：
        输入形状:
            query, key, value: (batch_size, n_local_heads, seqlen, head_dim)
        步骤:
        1) Q @ K^T 并除以 sqrt(head_dim) 得到注意力分数
        2) 加因果 mask，禁止看到未来位置
        3) softmax 得到注意力权重
        4) 对注意力权重做 dropout
        5) 用注意力权重加权求和 V，得到输出
        '''
        batch_size, n_heads, seqlen, head_dim = query.size()

        # 1) 计算注意力分数: (b, h, t_q, t_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        # 缩放
        attn_scores = attn_scores / math.sqrt(head_dim)

        # 2) 因果 mask：只允许注意当前以及之前的 token
        causal_mask = torch.tril(
            torch.ones((seqlen, seqlen), device=query.device, dtype=torch.bool)
        )  # (seqlen, seqlen)
        causal_mask = causal_mask.view(1, 1, seqlen, seqlen)  # 广播到 (b, h, t_q, t_k)
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # 3) softmax 得到注意力概率
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 4) dropout
        attn_probs = self.attn_dropout(attn_probs)

        # 5) 加权求和 V，得到输出: (b, h, seqlen, head_dim)
        output = torch.matmul(attn_probs, value)

        return output


    def forward(
        self,
        x: torch.Tensor
    ):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        '''
        Transformer Block 前向流程（Pre-LN 版本）：
        1) 对输入做 RMSNorm
        2) 对归一化后的表示做自注意力
        3) 残差连接：x + attn_output
        4) 对残差结果做 RMSNorm
        5) 送入前馈网络（FFN）
        6) 再做一次残差连接：h + ffn_output
        '''
        # 1) 归一化后送入注意力
        attn_input = self.attention_norm(x)
        # 2) 自注意力
        attn_output = self.attention(attn_input)
        # 3) 第一条残差连接
        h = x + attn_output

        # 4) 对 h 做归一化，送入 FFN
        ffn_input = self.ffn_norm(h)
        # 5) 前馈网络
        ffn_output = self.feed_forward(ffn_input)
        # 6) 第二条残差连接
        out = h + ffn_output

        return out



class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        自回归文本生成：
        - 输入 idx: (batch_size, cur_len) 的 token 序列
        - 每一步：
          1) 截断到 max_seq_len
          2) 前向得到最后一个时间步的 logits
          3) 根据 temperature 采样/选择下一个 token
          4) 拼接到序列末尾
        """
        for _ in range(max_new_tokens):
            # 1) 防止上下文长度超过最大长度
            if idx.size(1) > self.params.max_seq_len:
                idx_cond = idx[:, -self.params.max_seq_len:]
            else:
                idx_cond = idx

            # 2) 前向计算，拿到 logits
            logits, _ = self(idx_cond)       # (b, t, vocab_size)
            logits = logits[:, -1, :]        # 只取最后一个时间步 (b, vocab_size)

            if temperature == 0.0:
                # 温度为 0：贪心解码，取概率最大的 token
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (b, 1)
            else:
                '''
                Temperature sampling 过程：
                1) logits / temperature，温度越低分布越尖锐
                2) softmax 得到概率分布
                3) 从该分布中采样下一个 token
                '''
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)     # (b, 1)

            # 3) 把新 token 拼到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)

        return idx




def load_pretrained(checkpoint):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
  dtype = "float32"

  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  # init from a model saved in a specific directory
  checkpoint_dict = torch.load(checkpoint, map_location=device)
  config = LlamaConfig(**checkpoint_dict['model_args'])
  model = Llama(config)
  state_dict = checkpoint_dict['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  return model
