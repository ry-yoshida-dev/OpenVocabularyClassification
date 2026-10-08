import pytest
import torch
from transformers import Siglip2VisionConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead
from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead

from open_vocabulary_classification.backends.siglip import UniformAttentionPooling

HIDDEN_SIZE: int = 32


def make_attention_uniform(head: SiglipMultiheadAttentionPoolingHead | Siglip2MultiheadAttentionPoolingHead) -> None:
    with torch.no_grad():
        head.attention.in_proj_weight[HIDDEN_SIZE : 2 * HIDDEN_SIZE].zero_()
        head.attention.in_proj_bias[HIDDEN_SIZE : 2 * HIDDEN_SIZE].zero_()


def test_uniform_pooling_equals_the_head_with_uniform_attention() -> None:
    torch.manual_seed(0)
    head: SiglipMultiheadAttentionPoolingHead = SiglipMultiheadAttentionPoolingHead(
        SiglipVisionConfig(hidden_size=HIDDEN_SIZE, intermediate_size=37, num_attention_heads=4)
    ).eval()
    tokens: torch.Tensor = torch.randn(3, 5, HIDDEN_SIZE)
    learned_pooling: torch.Tensor = head(tokens)
    assert not torch.allclose(UniformAttentionPooling.pool(head, tokens, None), learned_pooling, atol=1e-4)
    make_attention_uniform(head)
    torch.testing.assert_close(UniformAttentionPooling.pool(head, tokens, None), head(tokens), atol=1e-5, rtol=1e-5)


def test_uniform_pooling_skips_padding() -> None:
    torch.manual_seed(0)
    config: Siglip2VisionConfig = Siglip2VisionConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=37, num_attention_heads=4
    )
    head: Siglip2MultiheadAttentionPoolingHead = Siglip2MultiheadAttentionPoolingHead(config).eval()
    make_attention_uniform(head)
    tokens: torch.Tensor = torch.randn(2, 6, HIDDEN_SIZE)
    patch_mask: torch.Tensor = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    pooled: torch.Tensor = UniformAttentionPooling.pool(head, tokens, patch_mask.bool())
    torch.testing.assert_close(pooled[1], head(tokens[1:])[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        pooled[0], UniformAttentionPooling.pool(head, tokens[:1, :4], None)[0], atol=1e-5, rtol=1e-5
    )


def test_image_without_patches_raises() -> None:
    head: SiglipMultiheadAttentionPoolingHead = SiglipMultiheadAttentionPoolingHead(
        SiglipVisionConfig(hidden_size=HIDDEN_SIZE, intermediate_size=37, num_attention_heads=4)
    )
    with pytest.raises(ValueError, match="at least one patch"):
        UniformAttentionPooling.pool(head, torch.randn(1, 3, HIDDEN_SIZE), torch.zeros(1, 3, dtype=torch.bool))
