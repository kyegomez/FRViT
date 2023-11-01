import torch
from torch import nn
import math


class MHA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py




    # TODO:
    - Sparsification
    - qk_norm
    - qlora linear

    outputs returned with them on are bogus


    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        stride: int = 32,
        expressivity: int = 8,
        is_bidirectional: bool = False,
        is_sparse: bool = False,
    ) -> None:
        """Build MHA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv = QloraLinear(dim, dim * 3)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        # self.proj = QloraLinear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        
        # Normalization
        self.qk_norm = qk_norm
        self.norm = nn.LayerNorm(head_dim)

        # Sparse Attn Params
        self.is_bidirectional = is_bidirectional
        self.stride = stride
        self.expressivity = expressivity
        assert self.stride > 0 and self.stride >= self.expressivity
        self.is_sparse = is_sparse

    def compute_checkpoints(self, word_index):
        if word_index % self.stride == 0 and word_index != 0:
            checkpoint_index = word_index - self.expressivity
        else:
            checkpoint_index = (
                math.floor(word_index / self.stride) * self.stride
                + self.stride
                - self.expressivity
            )
        return checkpoint_index
    
    def compute_subset_summaries(self, absolute_max):
        checkpoint_index = self.compute_checkpoints(0)
        subset_two = set()
        while checkpoint_index <= absolute_max - 1:
            summary = set(
                range(
                    checkpoint_index,
                    min(checkpoint_index + self.expressivity + 1, absolute_max),
                )
            )
            subset_two = subset_two.union(summary)
            checkpoint_index = self.compute_checkpoints(checkpoint_index + self.stride)
        return subset_two
    
    # Sparse Transformer Fixed Attention Pattern: https://arxiv.org/pdf/1904.10509.pdf
    def compute_fixed_attention_subset(self, word_index, tgt_len):
        # +1s account for range function; [min, max) -> [min, max]
        if not self.is_bidirectional:
            absolute_max = word_index + 1
        else:
            absolute_max = tgt_len

        # Subset 1 - whole window
        rounded_index = (
            math.floor((word_index + self.stride) / self.stride) * self.stride
        )
        if word_index % self.stride == 0 and word_index != 0:
            subset_one = set(
                range(word_index - self.stride, min(absolute_max, word_index + 1))
            )
        else:
            subset_one = set(
                range(
                    max(0, rounded_index - self.stride),
                    min(absolute_max, rounded_index + 1),
                )
            )

        # Subset 2 - summary per window
        # If bidirectional, subset 2 is the same for every index
        subset_two = set()
        if not self.is_bidirectional:
            subset_two = self.compute_subset_summaries(absolute_max)

        return subset_one.union(subset_two)

    # Compute sparse mask - if bidirectional, can pre-compute and store
    def buffered_sparse_mask(self, tensor, tgt_len, src_len):
        assert tgt_len > self.stride
        sparse_mask = torch.empty((tgt_len, src_len)).float().fill_(float("-inf"))

        # If bidirectional, subset 2 is the same for every index
        subset_summaries = set()
        if self.is_bidirectional:
            subset_summaries = self.compute_subset_summaries(tgt_len)

        for i in range(tgt_len):
            fixed_attention_subset = self.compute_fixed_attention_subset(i, tgt_len)
            fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
            included_word_indices = torch.LongTensor(list(fixed_attention_subset))
            sparse_mask[i].index_fill_(0, included_word_indices, 0)
        return sparse_mask.type_as(tensor)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        sparse_mask = self.buffered_sparse_mask(attn_weights, tgt_len, src_len)
        sparse_mask = sparse_mask.unsqueeze(0).expand(
            bsz * self.num_heads, tgt_len, src_len
        )

        # Print shapes
        print("Attn Weights Shape: ", attn_weights.shape)
        print("Sparse Mask Shape: ", sparse_mask.shape)

        # Reshape attn weights to match the shape of sparse mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        attn_weights += sparse_mask

        # # Reshape attn weights back to original shape
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W

        # LayerNorm before self attention
        # x = self.norm(x)

        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)


        # qk normalization
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
        

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Apply sparse mask to attn weights
        if self.is_sparse:
            self.apply_sparse_mask(attn, N, N, B)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


# Create a random 4D tensor
# Let's assume:
# Batch size (B) = 8
# Channels (C) = 128
# Height (H) = 32
# Width (W) = 32
x = torch.randn(8, 128, 32, 32)

# Initialize the MHA module
mha = MHA(dim=128, head_dim=32, qk_norm=False)

# Pass the random tensor through the MHA module
y = mha(x)

# Print the output shape
print(y)  # Expected: torch.Size([8, 128, 32, 32])