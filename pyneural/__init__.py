"""
PyNeural - Python Bindings for Neural Assembly Framework

A high-performance deep learning framework implemented in x86-64 assembly
with Python bindings via ctypes.

Example Usage:
    >>> import pyneural as pn
    >>> 
    >>> # Initialize the framework
    >>> pn.init()
    >>> 
    >>> # Create tensors
    >>> a = pn.Tensor.zeros([2, 3])
    >>> b = pn.Tensor.ones([2, 3])
    >>> 
    >>> # Perform operations
    >>> c = a + b
    >>> print(c.numpy())
    >>> 
    >>> # Create neural network layers
    >>> linear = pn.Linear(10, 5)
    >>> output = linear(input_tensor)
    >>> 
    >>> # Cleanup
    >>> pn.shutdown()
"""

from .tensor import Tensor
from .nn import (
    Linear, ReLU, Sigmoid, Softmax, Tanh, Dropout, Sequential,
    BatchNorm1d, LayerNorm, Embedding,
    Flatten, ResidualBlock,
    MSELoss, CrossEntropyLoss, LabelSmoothingCrossEntropy,
)
from .optim import SGD, Adam, AdamW
from .autograd import (
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
    inference_mode,
    register_hook,
    remove_hook,
    clear_hooks,
)
from .core import (
    init,
    shutdown,
    version,
    get_last_error,
    get_error_message,
    clear_error,
    get_simd_level,
    get_simd_name,
)
from .dataset import (
    Dataset,
    DataLoader,
    TensorDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
    BatchSampler,
)
from .config import Config

# New modules
from .metrics import ConfusionMatrix, compute_accuracy, top_k_accuracy, roc_auc_score
from .metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score, explained_variance,
)
from .schedulers import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    WarmupLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LRFinder,
)
from .training import Trainer, TrainerConfig, TrainingHistory, EarlyStopping, NaNDetector
from .checkpoint import save_checkpoint, load_checkpoint
from .tb_logger import SummaryWriter
from .pruning import Pruner, prune_magnitude, prune_rows, prune_cols
from .quantize import Quantizer, QuantParams, quantized_matmul
from .transforms import Normalize, MinMaxScale, Compose, compute_stats
from .transforms import (
    ToTensor, RandomHorizontalFlip, RandomVerticalFlip,
    RandomCrop, RandomRotation, RandomNoise, RandomErasing,
)
from .datasets import (
    MNIST, FashionMNIST, CIFAR10,
    Iris, WineQuality,
    VisionDataset, TabularDataset,
    train_test_split,
)
from .arch import parse_architecture, parse_residual_block
from .sklearn_compat import MLPClassifier, MLPRegressor
from .export import ONNXExporter, export_model
from .jupyter_utils import (
    ProgressBar, NotebookCallback, LivePlotter, EpochTimer,
    NotebookTrainer, configure_notebook_mode, setup_training_display
)
from .errors import (
    ValidationError, ShapeMismatchError, ConfigError,
    ErrorMessageBuilder, ShapeValidator, ConfigValidator, DataValidator,
    safe_shape_check
)
from .fuzzy import (
    FuzzySystem,
    triangular, trapezoidal, gaussian,
    fuzzy_and, fuzzy_or, fuzzy_not,
    defuzz_centroid, defuzz_bisector, defuzz_mom,
)
from .conv import Conv2D, MaxPool2D, calc_output_size
from .pooling import AvgPool2D, Upsample2D
from .tensor_ops import concat_2d, split_2d, pad_2d, transpose_2d
from .attention import scaled_dot_product_attention, transformer_block
from .activations import (
    GELU, LeakyReLU, ELU, SELU, Swish, Mish, HardSwish, Softplus, HardTanh,
)
from .rnn import LSTM, GRU

import importlib as _importlib
weight_init = _importlib.import_module('.init', __name__)

# Re-import core.init to ensure pn.init() is the framework initializer
from .core import init

__version__ = "3.0.0"
__all__ = [
    # Core
    "init",
    "shutdown", 
    "version",
    "get_last_error",
    "get_error_message",
    "clear_error",
    "get_simd_level",
    "get_simd_name",
    # Tensor
    "Tensor",
    # Neural Network
    "Linear",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "Dropout",
    "Sequential",
    "Embedding",
    "Flatten",
    "ResidualBlock",
    # Normalization
    "BatchNorm1d",
    "LayerNorm",
    # Loss Functions
    "MSELoss",
    "CrossEntropyLoss",
    "LabelSmoothingCrossEntropy",
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    # Autograd
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "inference_mode",
    "register_hook",
    "remove_hook",
    "clear_hooks",
    # Data
    "Dataset",
    "DataLoader",
    "TensorDataset",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "WeightedRandomSampler",
    "BatchSampler",
    "Config",
    # Native Datasets
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "Iris",
    "WineQuality",
    "VisionDataset",
    "TabularDataset",
    "train_test_split",
    # Metrics - Classification
    "ConfusionMatrix",
    "compute_accuracy",
    "top_k_accuracy",
    "roc_auc_score",
    # Metrics - Regression
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "explained_variance",
    # Schedulers
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "WarmupLR",
    "ReduceLROnPlateau",
    "OneCycleLR",
    "LRFinder",
    # Training
    "Trainer",
    "TrainerConfig",
    "TrainingHistory",
    "EarlyStopping",
    "NaNDetector",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    # Architecture DSL
    "parse_architecture",
    "parse_residual_block",
    # sklearn-compatible API
    "MLPClassifier",
    "MLPRegressor",
    # Weight Initialization
    "weight_init",
    # TensorBoard Logging
    "SummaryWriter",
    # Pruning
    "Pruner",
    "prune_magnitude",
    "prune_rows",
    "prune_cols",
    # Quantization
    "Quantizer",
    "QuantParams",
    "quantized_matmul",
    # Transforms
    "Normalize",
    "MinMaxScale",
    "Compose",
    "compute_stats",
    "ToTensor",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomCrop",
    "RandomRotation",
    "RandomNoise",
    "RandomErasing",
    # Fuzzy Logic
    "FuzzySystem",
    "triangular",
    "trapezoidal",
    "gaussian",
    "fuzzy_and",
    "fuzzy_or",
    "fuzzy_not",
    "defuzz_centroid",
    "defuzz_bisector",
    "defuzz_mom",
    # Convolution / Pooling
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "Upsample2D",
    "calc_output_size",
    # Tensor ops
    "concat_2d",
    "split_2d",
    "pad_2d",
    "transpose_2d",
    # Attention / Transformer
    "scaled_dot_product_attention",
    "transformer_block",
    # Extended Activations
    "GELU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "Swish",
    "Mish",
    "HardSwish",
    "Softplus",
    "HardTanh",
    # Recurrent Layers
    "LSTM",
    "GRU",
    # ONNX Export
    "ONNXExporter",
    "export_model",
    # Jupyter Integration
    "ProgressBar",
    "NotebookCallback",
    "LivePlotter",
    "EpochTimer",
    "NotebookTrainer",
    "configure_notebook_mode",
    "setup_training_display",
    # Error Handling
    "ValidationError",
    "ShapeMismatchError",
    "ConfigError",
    "ErrorMessageBuilder",
    "ShapeValidator",
    "ConfigValidator",
    "DataValidator",
    "safe_shape_check",
]
