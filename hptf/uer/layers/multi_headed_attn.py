import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale = True,
                 use_rope=False, rope_base=10000.0):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.inner_hidden_size = heads_num * attention_head_size

        if self.use_rope and self.per_head_size % 2 != 0:
            raise ValueError("RoPE requires an even attention head size.")

        self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def _rope_cache(self, seq_length, device, dtype):
        inv_freq = 1.0 / (
            self.rope_base ** (
                torch.arange(0, self.per_head_size, 2, device=device, dtype=torch.float32) / self.per_head_size
            )
        )
        positions = torch.arange(seq_length, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        x_1 = x[..., ::2]
        x_2 = x[..., 1::2]
        rotated = torch.stack((-x_2, x_1), dim=-1)
        return rotated.flatten(start_dim=-2)

    def _apply_rope(self, x):
        cos, sin = self._rope_cache(x.size(-2), x.device, x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, key, value, query, mask, position_bias=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, self.inner_hidden_size)


        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]

        if self.use_rope:
            query = self._apply_rope(query)
            key = self._apply_rope(key)

        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output
