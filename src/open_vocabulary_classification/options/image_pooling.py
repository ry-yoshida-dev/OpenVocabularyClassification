from enum import StrEnum


class ImagePooling(StrEnum):
    """
    How the token sequence of the vision encoder is reduced to one image embedding in the joint image-text space.

    Every pooling ends in the model's own projection, so text queries remain comparable with the image embedding.
    Which poolings a model offers depends on its family (``ClassifierBackend.supported_image_poolings``).

    Attributes
    ----------
    CLASS_TOKEN : str
        The class (CLS) token, normalized by the final layer norm and projected; how CLIP was trained.
    ATTENTION : str
        Multihead attention pooling (MAP) of every patch token by a learned probe; how SigLIP was trained.
    PATCH_MEAN : str
        Global average pooling: the mean of every patch token (class token excluded, padding of variable-resolution
        inputs excluded). CLIP averages the normalized patch tokens and projects the mean; SigLIP sends the mean
        through its pooling head with uniform instead of learned attention weights. It weighs every region equally,
        which helps when the object is small or off-center, but the models were not trained to use it.
    """

    CLASS_TOKEN = "class_token"
    ATTENTION = "attention"
    PATCH_MEAN = "patch_mean"
