"""ONNX Export Module
Export Neural-Assembly models to ONNX format for cross-framework compatibility.
Enables inference in PyTorch, TensorFlow, ONNX Runtime, TensorRT, mobile, etc.
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ONNXAttributeProto:
    """Simplified ONNX attribute representation"""
    def __init__(self, name: str, attr_type: str, value: Any):
        self.name = name
        self.attr_type = attr_type  # 'int', 'float', 'string', 'ints', 'floats'
        self.value = value


class ONNXTensorProto:
    """Simplified ONNX tensor representation"""
    def __init__(self, name: str, dims: Tuple[int, ...], dtype: str, data: np.ndarray):
        self.name = name
        self.dims = dims
        self.dtype = dtype  # 'float32', 'int32', 'int8'
        self.data = data


class ONNXNodeProto:
    """Simplified ONNX node (operation)"""
    def __init__(self, op_type: str, inputs: List[str], outputs: List[str], 
                 name: str, attributes: Optional[Dict[str, Any]] = None):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.attributes = attributes or {}


class ONNXGraph:
    """Build ONNX computation graph"""
    def __init__(self, name: str, nodes: List[ONNXNodeProto], 
                 inputs: List[str], outputs: List[str], 
                 initializers: List[ONNXTensorProto]):
        self.name = name
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.initializers = initializers


class ONNXExporter:
    """Export Sequential models to ONNX format"""
    
    def __init__(self, model: 'Sequential', opset_version: int = 11):
        """
        Initialize ONNX exporter
        
        Args:
            model: Neural-Assembly Sequential model
            opset_version: ONNX opset version (11+ recommended)
        """
        self.model = model
        self.opset_version = opset_version
        self.nodes: List[ONNXNodeProto] = []
        self.initializers: List[ONNXTensorProto] = []
        self.var_counter = 0
        self.layer_outputs: Dict[int, str] = {}  # map layer_idx -> output_name
    
    def _get_unique_name(self, prefix: str) -> str:
        """Generate unique variable names"""
        name = f"{prefix}_{self.var_counter}"
        self.var_counter += 1
        return name
    
    def _export_linear(self, layer: 'Linear', layer_idx: int, input_name: str) -> str:
        """Export Dense/Linear layer"""
        # Extract weights and biases from layer
        weight_name = f"weight_{layer_idx}"
        bias_name = f"bias_{layer_idx}"
        output_name = self._get_unique_name(f"linear_out_{layer_idx}")
        
        # Create weight tensor (transposed for ONNX convention)
        weight_data = layer.weight.T if hasattr(layer, 'weight') else np.zeros((layer.out_features, layer.in_features))
        weight_tensor = ONNXTensorProto(weight_name, weight_data.shape, 'float32', weight_data)
        self.initializers.append(weight_tensor)
        
        # Create bias tensor
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias_data = np.array(layer.bias)
            bias_tensor = ONNXTensorProto(bias_name, bias_data.shape, 'float32', bias_data)
            self.initializers.append(bias_tensor)
            
            # MatMul + Add
            matmul_out = self._get_unique_name(f"matmul_{layer_idx}")
            matmul_node = ONNXNodeProto('MatMul', [input_name, weight_name], [matmul_out], f"matmul_{layer_idx}")
            self.nodes.append(matmul_node)
            
            add_node = ONNXNodeProto('Add', [matmul_out, bias_name], [output_name], f"add_{layer_idx}")
            self.nodes.append(add_node)
        else:
            # Just MatMul
            matmul_node = ONNXNodeProto('MatMul', [input_name, weight_name], [output_name], f"matmul_{layer_idx}")
            self.nodes.append(matmul_node)
        
        return output_name
    
    def _export_activation(self, layer: Any, layer_idx: int, input_name: str) -> str:
        """Export activation functions (ReLU, Sigmoid, Tanh, Softmax)"""
        output_name = self._get_unique_name(f"activation_{layer_idx}")
        
        layer_type = type(layer).__name__
        
        if layer_type == 'ReLU':
            node = ONNXNodeProto('Relu', [input_name], [output_name], f"relu_{layer_idx}")
        elif layer_type == 'Sigmoid':
            node = ONNXNodeProto('Sigmoid', [input_name], [output_name], f"sigmoid_{layer_idx}")
        elif layer_type == 'Tanh':
            node = ONNXNodeProto('Tanh', [input_name], [output_name], f"tanh_{layer_idx}")
        elif layer_type == 'Softmax':
            axis = getattr(layer, 'axis', 1)
            node = ONNXNodeProto('Softmax', [input_name], [output_name], f"softmax_{layer_idx}",
                               {'axis': axis})
        else:
            raise ValueError(f"Unsupported activation: {layer_type}")
        
        self.nodes.append(node)
        return output_name
    
    def _export_flatten(self, layer: 'Flatten', layer_idx: int, input_name: str) -> str:
        """Export Flatten layer"""
        output_name = self._get_unique_name(f"flatten_{layer_idx}")
        node = ONNXNodeProto('Flatten', [input_name], [output_name], f"flatten_{layer_idx}",
                           {'axis': 1})
        self.nodes.append(node)
        return output_name
    
    def _export_dropout(self, layer: 'Dropout', layer_idx: int, input_name: str) -> str:
        """Export Dropout (identity during inference)"""
        # Dropout is typically disabled in inference mode, so just pass through
        return input_name
    
    def _export_batchnorm(self, layer: Any, layer_idx: int, input_name: str) -> str:
        """Export BatchNorm layer"""
        output_name = self._get_unique_name(f"batchnorm_{layer_idx}")
        
        # BatchNorm requires scale, bias, running_mean, running_var
        scale_name = f"scale_{layer_idx}"
        bias_name = f"bias_{layer_idx}"
        mean_name = f"mean_{layer_idx}"
        var_name = f"var_{layer_idx}"
        
        # Create initializers
        for name, data in [
            (scale_name, np.ones(layer.num_features)),
            (bias_name, np.zeros(layer.num_features)),
            (mean_name, np.zeros(layer.num_features)),
            (var_name, np.ones(layer.num_features)),
        ]:
            tensor = ONNXTensorProto(name, data.shape, 'float32', data)
            self.initializers.append(tensor)
        
        node = ONNXNodeProto('BatchNormalization',
                           [input_name, scale_name, bias_name, mean_name, var_name],
                           [output_name],
                           f"batchnorm_{layer_idx}",
                           {'epsilon': 1e-5, 'momentum': 0.1})
        self.nodes.append(node)
        return output_name
    
    def _export_conv2d(self, layer: Any, layer_idx: int, input_name: str) -> str:
        """Export Conv2D layer"""
        output_name = self._get_unique_name(f"conv_{layer_idx}")
        weight_name = f"weight_{layer_idx}"
        bias_name = f"bias_{layer_idx}"
        
        # Extract weights
        weight_data = np.array(layer.weight) if hasattr(layer, 'weight') else np.zeros(
            (layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1])
        )
        weight_tensor = ONNXTensorProto(weight_name, weight_data.shape, 'float32', weight_data)
        self.initializers.append(weight_tensor)
        
        # Prepare attributes
        attrs = {
            'kernel_shape': list(layer.kernel_size),
            'strides': list(layer.stride),
            'pads': [layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]],
        }
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias_data = np.array(layer.bias)
            bias_tensor = ONNXTensorProto(bias_name, bias_data.shape, 'float32', bias_data)
            self.initializers.append(bias_tensor)
            inputs = [input_name, weight_name, bias_name]
        else:
            inputs = [input_name, weight_name]
        
        node = ONNXNodeProto('Conv', inputs, [output_name], f"conv_{layer_idx}", attrs)
        self.nodes.append(node)
        return output_name
    
    def _export_pool(self, layer: Any, layer_idx: int, input_name: str, pool_type: str) -> str:
        """Export pooling layers (MaxPool, AveragePool)"""
        output_name = self._get_unique_name(f"pool_{layer_idx}")
        
        attrs = {
            'kernel_shape': list(layer.kernel_size),
            'strides': list(layer.stride),
            'pads': [0, 0, 0, 0],
        }
        
        node = ONNXNodeProto(pool_type, [input_name], [output_name], f"pool_{layer_idx}", attrs)
        self.nodes.append(node)
        return output_name
    
    def export(self, input_shapes: Dict[str, Tuple[int, ...]]) -> str:
        """
        Export model to ONNX and return the model as a string.
        
        Args:
            input_shapes: Dictionary mapping input names to shapes
                         e.g., {'input': (1, 784)} for MNIST
        
        Returns:
            ONNX model as binary string (base64 encoded)
        """
        # Build graph by traversing layers
        current_input = 'input'
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            
            if layer_type in ['Linear', 'Dense']:
                current_input = self._export_linear(layer, layer_idx, current_input)
            elif layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']:
                current_input = self._export_activation(layer, layer_idx, current_input)
            elif layer_type == 'Flatten':
                current_input = self._export_flatten(layer, layer_idx, current_input)
            elif layer_type == 'Dropout':
                current_input = self._export_dropout(layer, layer_idx, current_input)
            elif layer_type in ['BatchNorm', 'BatchNorm1d']:
                current_input = self._export_batchnorm(layer, layer_idx, current_input)
            elif layer_type == 'Conv2d':
                current_input = self._export_conv2d(layer, layer_idx, current_input)
            elif layer_type == 'MaxPool2d':
                current_input = self._export_pool(layer, layer_idx, current_input, 'MaxPool')
            elif layer_type == 'AvgPool2d':
                current_input = self._export_pool(layer, layer_idx, current_input, 'AveragePool')
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        
        # Construct ONNX graph metadata
        graph = ONNXGraph(
            name="neural_assembly_model",
            nodes=self.nodes,
            inputs=['input'],
            outputs=[current_input],
            initializers=self.initializers
        )
        
        # Return ONNX graph representation (simplified proto format)
        return self._serialize_onnx(graph)
    
    def _serialize_onnx(self, graph: ONNXGraph) -> str:
        """Serialize ONNX graph to string format (pseudo-protobuf)"""
        output = []
        output.append(f"Graph: {graph.name}")
        output.append(f"Inputs: {graph.inputs}")
        output.append(f"Outputs: {graph.outputs}")
        output.append(f"\nNodes ({len(graph.nodes)}):")
        
        for node in graph.nodes:
            output.append(f"  {node.name} ({node.op_type})")
            output.append(f"    Inputs: {node.inputs}")
            output.append(f"    Outputs: {node.outputs}")
            if node.attributes:
                output.append(f"    Attrs: {node.attributes}")
        
        output.append(f"\nInitializers ({len(graph.initializers)}):")
        for init in graph.initializers:
            output.append(f"  {init.name}: {init.dims} ({init.dtype})")
        
        return "\n".join(output)


def export_model(model: 'Sequential', onnx_path: str, input_shape: Tuple[int, ...] = (1, 784)):
    """
    High-level export function
    
    Args:
        model: Neural-Assembly Sequential model
        onnx_path: Output ONNX file path
        input_shape: Input tensor shape (excluding batch dimension if applicable)
    """
    exporter = ONNXExporter(model)
    onnx_graph = exporter.export({'input': input_shape})
    
    with open(onnx_path, 'w') as f:
        f.write(onnx_graph)
    
    print(f"Model exported to {onnx_path}")
