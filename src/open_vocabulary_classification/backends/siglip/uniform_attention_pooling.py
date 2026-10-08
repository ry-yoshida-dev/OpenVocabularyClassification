import torch
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead
from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead


class UniformAttentionPooling:
    """
    Global average pooling of SigLIP patch tokens, carried into the joint space by the attention pooling head.

    The head of SigLIP (multihead attention pooling, MAP) attends from a learned probe to every patch token and
    refines the result with a residual MLP; its output is the image embedding. Replacing the learned attention weights
    with uniform weights over the patches turns the attention into the value and output projections of the mean patch
    token, because both projections are affine and the weights of every head sum to one. The residual MLP then runs
    unchanged, so the mean token lands in the space text queries are embedded in.
    """

    @staticmethod
    def pool(
        head: SiglipMultiheadAttentionPoolingHead | Siglip2MultiheadAttentionPoolingHead,
        tokens: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Pool the patch tokens with uniform attention weights.

        Parameters
        ----------
        head : SiglipMultiheadAttentionPoolingHead | Siglip2MultiheadAttentionPoolingHead
            Attention pooling head of the vision tower.
        tokens : torch.Tensor
            Patch tokens after the final layer norm, shape (B, P, H).
        patch_mask : torch.Tensor | None
            Whether each token is an image patch rather than padding, shape (B, P); ``None`` averages every token.

        Returns
        -------
        torch.Tensor
            Pooled image embeddings, shape (B, H).

        Raises
        ------
        ValueError
            If an image has no patch in ``patch_mask``.
        RuntimeError
            If the attention of the head keeps separate query, key and value weights.
        """
        patch_weights: torch.Tensor = (
            torch.ones(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)
            if patch_mask is None
            else patch_mask.to(device=tokens.device, dtype=tokens.dtype)
        )
        patch_counts: torch.Tensor = patch_weights.sum(dim=1, keepdim=True)
        if bool((patch_counts == 0).any()):
            raise ValueError("every image needs at least one patch in patch_mask.")
        mean_tokens: torch.Tensor = (tokens * patch_weights.unsqueeze(-1)).sum(dim=1) / patch_counts
        attention: torch.nn.MultiheadAttention = head.attention
        if attention.kdim != attention.embed_dim or attention.vdim != attention.embed_dim:
            raise RuntimeError("the pooling attention must pack its query, key and value weights into one matrix.")
        value_rows: slice = slice(2 * attention.embed_dim, 3 * attention.embed_dim)
        values: torch.Tensor = torch.nn.functional.linear(
            mean_tokens, attention.in_proj_weight[value_rows], attention.in_proj_bias[value_rows]
        )
        attended: torch.Tensor = attention.out_proj(values)
        pooled: torch.Tensor = attended + head.mlp(head.layernorm(attended))
        return pooled
