"""
Configuration utilities for PyNeural.
"""

import ctypes
from pathlib import Path
from typing import Optional, Union

from .core import _lib, NeuralException


class Config:
    """
    Configuration loader for INI-style config files.
    
    Example:
        >>> config = Config.load("config.ini")
        >>> lr = config.get_float("training", "learning_rate", default=0.01)
        >>> epochs = config.get_int("training", "epochs", default=100)
    """
    
    def __init__(self, ptr: ctypes.c_void_p):
        """
        Create a Config wrapper around a native config pointer.
        
        Args:
            ptr: Pointer to native NeuralConfig
        """
        if ptr is None or ptr == 0:
            raise NeuralException(1, "Null config pointer")
        self._ptr = ptr
    
    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.neural_config_free(self._ptr)
            self._ptr = None
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """
        Load configuration from an INI file.
        
        Args:
            path: Path to configuration file
        
        Returns:
            Config instance
        """
        path = str(Path(path).resolve())
        ptr = _lib.neural_config_load(path.encode("utf-8"))
        
        if ptr is None or ptr == 0:
            raise NeuralException(
                _lib.neural_get_last_error(),
                f"Failed to load config from {path}"
            )
        
        return cls(ptr)
    
    def get_int(
        self,
        section: str,
        key: str,
        default: int = 0
    ) -> int:
        """
        Get an integer value from the config.
        
        Args:
            section: Section name (e.g., "training")
            key: Key name (e.g., "epochs")
            default: Default value if not found
        
        Returns:
            Integer value
        """
        return _lib.neural_config_get_int(
            self._ptr,
            section.encode("utf-8"),
            key.encode("utf-8"),
            default
        )
    
    def get_float(
        self,
        section: str,
        key: str,
        default: float = 0.0
    ) -> float:
        """
        Get a float value from the config.
        
        Args:
            section: Section name
            key: Key name
            default: Default value if not found
        
        Returns:
            Float value
        """
        return _lib.neural_config_get_float(
            self._ptr,
            section.encode("utf-8"),
            key.encode("utf-8"),
            ctypes.c_double(default)
        )
    
    def get_string(
        self,
        section: str,
        key: str,
        default: str = ""
    ) -> str:
        """
        Get a string value from the config.
        
        Args:
            section: Section name
            key: Key name
            default: Default value if not found
        
        Returns:
            String value
        """
        result = _lib.neural_config_get_string(
            self._ptr,
            section.encode("utf-8"),
            key.encode("utf-8"),
            default.encode("utf-8")
        )
        if result:
            return result.decode("utf-8")
        return default
    
    def __repr__(self) -> str:
        return f"Config()"
