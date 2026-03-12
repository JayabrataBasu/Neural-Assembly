"""Enhanced Error Messages
Provides detailed, actionable error messages with suggestions for common mistakes
"""

import sys
from typing import Optional, Any, IO


class ValidationError(Exception):
    """Base validation error with helpful message"""
    pass


class ShapeMismatchError(ValidationError):
    """Tensor shape incompatibility"""
    pass


class ConfigError(ValidationError):
    """Configuration error"""
    pass


class ErrorMessageBuilder:
    """Build detailed error messages with context and suggestions"""
    
    def __init__(self, error_type: str, context: Optional[dict] = None):
        """
        Initialize error builder
        
        Args:
            error_type: Type of error (e.g., 'shape_mismatch', 'dtype_mismatch')
            context: Additional context (variable values, etc.)
        """
        self.error_type = error_type
        self.context = context or {}
        self.suggestions: list = []
    
    def add_suggestion(self, suggestion: str) -> 'ErrorMessageBuilder':
        """Add a suggestion to fix the error"""
        self.suggestions.append(suggestion)
        return self
    
    def build(self) -> str:
        """Build complete error message"""
        lines = []
        lines.append(f"❌ {self.error_type.upper()}")
        lines.append("")
        
        # Context information
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Suggestions
        if self.suggestions:
            lines.append("💡 Suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")
        
        return "\n".join(lines)


class ShapeValidator:
    """Validate tensor shapes with helpful errors"""
    
    @staticmethod
    def check_compatible(input_shape: tuple, expected_shape: tuple,
                        layer_name: str = "Unknown"):
        """
        Check if input shape matches expected shape
        
        Args:
            input_shape: Actual input shape
            expected_shape: Expected shape
            layer_name: Name of layer for error message
        
        Raises:
            ShapeMismatchError: If shapes don't match
        """
        if len(input_shape) != len(expected_shape):
            builder = ErrorMessageBuilder(
                "Shape Dimension Mismatch",
                {
                    'layer': layer_name,
                    'input_shape': input_shape,
                    'expected_shape': expected_shape,
                    'input_dims': len(input_shape),
                    'expected_dims': len(expected_shape)
                }
            )
            builder.add_suggestion(f"Input has {len(input_shape)} dimensions but {layer_name} expects {len(expected_shape)}")
            
            if layer_name == "Flatten":
                builder.add_suggestion("Flatten expects 4D input (batch, channels, height, width) before Conv layers")
            elif layer_name == "Linear":
                builder.add_suggestion("Linear layer expects 2D input (batch, features). Use Flatten first if input is 4D")
            
            raise ShapeMismatchError(builder.build())
        
        for i, (inp, exp) in enumerate(zip(input_shape, expected_shape)):
            if exp is not None and inp != exp:
                # Allow batch size -1 for flexible batching
                if i == 0 and exp == -1:
                    continue
                
                builder = ErrorMessageBuilder(
                    "Shape Size Mismatch",
                    {
                        'layer': layer_name,
                        'dimension': i,
                        'actual_size': inp,
                        'expected_size': exp,
                        'input_shape': input_shape,
                        'expected_shape': expected_shape
                    }
                )
                
                builder.add_suggestion(f"Dimension {i}: got {inp}, expected {exp}")
                
                if i == 1 and layer_name == "Linear":
                    builder.add_suggestion(f"Previous layer output {inp} doesn't match Linear input {exp}")
                    builder.add_suggestion("Check your architecture definition or previous layer configuration")
                
                raise ShapeMismatchError(builder.build())
    
    @staticmethod
    def check_2d(tensor: Any, name: str = "Tensor"):
        """Check if tensor is 2D"""
        shape = tensor.shape if hasattr(tensor, 'shape') else (len(tensor), len(tensor[0]))
        if len(shape) != 2:
            builder = ErrorMessageBuilder(
                "Tensor Dimension Error",
                {'tensor_name': name, 'shape': shape, 'dimensions': len(shape)}
            )
            builder.add_suggestion(f"{name} is {len(shape)}D, expected 2D (batch_size, features)")
            builder.add_suggestion(f"Try flattening with X.reshape(-1, {shape[-1] if len(shape) > 0 else 1})")
            raise ShapeMismatchError(builder.build())
    
    @staticmethod
    def check_4d_image(tensor: Any, name: str = "Tensor"):
        """Check if tensor is 4D (batch, channels, height, width)"""
        shape = tensor.shape if hasattr(tensor, 'shape') else None
        if shape is None or len(shape) != 4:
            builder = ErrorMessageBuilder(
                "Image Tensor Dimension Error",
                {'tensor_name': name, 'shape': shape, 'required_format': 'NCHW'}
            )
            builder.add_suggestion(f"{name} should be 4D (batch, channels, height, width)")
            builder.add_suggestion("If using PIL images, convert with numpy: np.array(image).transpose(2,0,1)")
            raise ShapeMismatchError(builder.build())


class ConfigValidator:
    """Validate configuration with helpful errors"""
    
    @staticmethod
    def check_range(value: float, min_val: float, max_val: float,
                   param_name: str):
        """Check if parameter is within valid range"""
        if not (min_val <= value <= max_val):
            builder = ErrorMessageBuilder(
                "Configuration Out of Range",
                {
                    'parameter': param_name,
                    'value': value,
                    'valid_range': f"[{min_val}, {max_val}]"
                }
            )
            builder.add_suggestion(f"Set {param_name} to a value between {min_val} and {max_val}")
            raise ConfigError(builder.build())
    
    @staticmethod
    def check_learning_rate(learning_rate: float):
        """Validate learning rate with common mistakes"""
        if learning_rate <= 0:
            builder = ErrorMessageBuilder(
                "Invalid Learning Rate",
                {'learning_rate': learning_rate}
            )
            builder.add_suggestion("Learning rate must be positive (> 0)")
            builder.add_suggestion("Typical values: 0.001, 0.01, 0.1")
            raise ConfigError(builder.build())
        
        if learning_rate > 1.0:
            builder = ErrorMessageBuilder(
                "Learning Rate Very Large",
                {'learning_rate': learning_rate, 'typical_max': 1.0}
            )
            builder.add_suggestion(f"Learning rate {learning_rate} is unusually large")
            builder.add_suggestion("Try reducing to 0.1 or smaller to prevent divergence")
            builder.add_suggestion("If intentional, this may cause training instability")
            print(builder.build())
    
    @staticmethod
    def check_batch_size(batch_size: int, dataset_size: int):
        """Validate batch size"""
        if batch_size <= 0:
            builder = ErrorMessageBuilder(
                "Invalid Batch Size",
                {'batch_size': batch_size}
            )
            builder.add_suggestion("Batch size must be positive")
            raise ConfigError(builder.build())
        
        if batch_size > dataset_size:
            builder = ErrorMessageBuilder(
                "Batch Size Exceeds Dataset",
                {
                    'batch_size': batch_size,
                    'dataset_size': dataset_size
                }
            )
            builder.add_suggestion(f"Batch size ({batch_size}) is larger than dataset ({dataset_size})")
            builder.add_suggestion(f"Use batch_size <= {dataset_size}")
            raise ConfigError(builder.build())


class DataValidator:
    """Validate input data"""
    
    @staticmethod
    def check_csv_encoding(file_path: str, default: str = 'utf-8') -> str:
        """
        Detect CSV encoding and suggest fixes
        
        Args:
            file_path: Path to CSV file
            default: Default encoding to try
        
        Returns:
            Detected encoding
        """
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Test read
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        builder = ErrorMessageBuilder(
            "CSV Encoding Error",
            {'file': file_path, 'attempted_encodings': encodings}
        )
        builder.add_suggestion(f"Cannot detect encoding for {file_path}")
        builder.add_suggestion("Try: df = pd.read_csv(filepath, encoding='latin-1')")
        builder.add_suggestion("Or: df = pd.read_csv(filepath, encoding='iso-8859-1')")
        raise ConfigError(builder.build())
    
    @staticmethod
    def check_label_distribution(labels: Any, warn_threshold: float = 0.9):
        """
        Check for class imbalance
        
        Args:
            labels: Array of labels
            warn_threshold: Warn if max class > threshold of total
        """
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        
        max_class_ratio = counts.max() / len(labels)
        
        if max_class_ratio > warn_threshold:
            builder = ErrorMessageBuilder(
                "Class Imbalance Detected",
                {
                    'max_class_ratio': f"{max_class_ratio*100:.1f}%",
                    'num_classes': len(unique),
                    'class_distribution': dict(zip(unique, counts.tolist()))
                }
            )
            builder.add_suggestion("Dataset is heavily imbalanced")
            builder.add_suggestion("Consider using class_weight='balanced' in training")
            builder.add_suggestion("Or use stratified cross-validation for fairer splits")
            print(builder.build())


def safe_shape_check(tensor: Any, expected: tuple, context: str = ""):
    """
    Safe shape check with helpful error
    
    Args:
        tensor: Input tensor
        expected: Expected shape
        context: Additional context for error message
    
    Raises:
        ShapeMismatchError: If shapes don't match
    """
    actual = tensor.shape if hasattr(tensor, 'shape') else ()
    
    if actual != expected:
        builder = ErrorMessageBuilder(
            "Shape Mismatch",
            {
                'context': context,
                'actual_shape': actual,
                'expected_shape': expected
            }
        )
        builder.add_suggestion(f"Got shape {actual}, expected {expected}")
        raise ShapeMismatchError(builder.build())
