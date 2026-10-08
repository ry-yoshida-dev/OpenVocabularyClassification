from enum import StrEnum


class Device(StrEnum):
    """
    Compute device a classifier runs on; each runtime maps it to its own device handle.

    Attributes
    ----------
    AUTO : str
        Best available device: CUDA, then Apple Silicon (MPS), then CPU.
    CPU : str
        CPU.
    CUDA : str
        Current CUDA GPU.
    MPS : str
        Apple Silicon GPU through Metal Performance Shaders.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
