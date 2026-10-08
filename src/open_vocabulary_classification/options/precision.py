from enum import StrEnum


class Precision(StrEnum):
    """
    Floating-point type the model weights and image inputs are run in; each runtime maps it to its own dtype.

    Embeddings and logits are always returned in float64, whatever the precision of the model.

    Attributes
    ----------
    FLOAT32 : str
        Single precision; supported on every device.
    FLOAT16 : str
        Half precision; GPU only. Halves memory and is fast on most GPUs, but large models may overflow.
    BFLOAT16 : str
        Brain floating point; the range of float32 with the memory of float16, so large models stay stable. Needs a
        GPU or CPU supporting it.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
