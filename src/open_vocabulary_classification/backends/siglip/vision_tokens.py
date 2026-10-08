from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class SiglipVisionTokens:
    """
    Output of a SigLIP vision tower needed by every pooling.

    Attributes
    ----------
    tokens : torch.Tensor
        Patch tokens after the final layer norm, shape (B, P, H).
    attention_pooled : torch.Tensor
        Output of the attention pooling head, the image embedding SigLIP was trained with, shape (B, H).
    patch_mask : torch.Tensor | None
        Whether each token is an image patch rather than padding, shape (B, P); ``None`` when every token is a patch.

    Raises
    ------
    ValueError
        If the shapes do not agree.
    """

    tokens: torch.Tensor
    attention_pooled: torch.Tensor
    patch_mask: torch.Tensor | None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B, P, H). got {tuple(self.tokens.shape)}")
        if self.attention_pooled.shape != (self.tokens.shape[0], self.tokens.shape[2]):
            raise ValueError(
                f"attention_pooled must have shape {(self.tokens.shape[0], self.tokens.shape[2])}. "
                + f"got {tuple(self.attention_pooled.shape)}"
            )
        if self.patch_mask is not None and self.patch_mask.shape != self.tokens.shape[:2]:
            raise ValueError(
                f"patch_mask must have shape {tuple(self.tokens.shape[:2])}. got {tuple(self.patch_mask.shape)}"
            )
