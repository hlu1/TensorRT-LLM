from abc import ABC, abstractmethod

import torch
from torch import nn


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(self, module: torch.nn.Module, *args, **kwargs):
        """Create weights for a module and set them as attributes of the module."""

        raise NotImplementedError

    @abstractmethod
    def load_weights(self, module: torch.nn.Module, *args, **kwargs):
        """Load weights for a module."""
        raise NotImplementedError

    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Process the weight after loading.
        Useful for transposing or shuffling weights.
        """
        return

    @abstractmethod
    def apply(self, module: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in module to the input tensor.

        Both create_weights and load_weights must have been called before on the module."""
        raise NotImplementedError
