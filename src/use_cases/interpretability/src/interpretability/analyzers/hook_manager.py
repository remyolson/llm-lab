"""Hook management system for feature extraction from neural networks."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookOutput:
    """Container for hook outputs."""

    layer_name: str
    module_type: str
    input_tensors: Optional[Tuple[torch.Tensor, ...]] = None
    output_tensors: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
    gradients: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookManager:
    """Manages hooks for extracting features from neural network models."""

    def __init__(self, model: nn.Module):
        """
        Initialize the hook manager.

        Args:
            model: The PyTorch model to attach hooks to
        """
        self.model = model
        self.hooks = []
        self.hook_outputs = defaultdict(list)
        self._hook_enabled = True

    def register_forward_hook(
        self, module: nn.Module, layer_name: str, hook_fn: Optional[Callable] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a forward hook on a module.

        Args:
            module: Module to attach hook to
            layer_name: Name identifier for the layer
            hook_fn: Optional custom hook function

        Returns:
            Removable hook handle
        """
        if hook_fn is None:
            hook_fn = self._create_forward_hook(layer_name)

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def register_backward_hook(
        self, module: nn.Module, layer_name: str, hook_fn: Optional[Callable] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a backward hook on a module.

        Args:
            module: Module to attach hook to
            layer_name: Name identifier for the layer
            hook_fn: Optional custom hook function

        Returns:
            Removable hook handle
        """
        if hook_fn is None:
            hook_fn = self._create_backward_hook(layer_name)

        handle = module.register_backward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def register_hooks_by_type(self, module_types: List[type], hook_type: str = "forward"):
        """
        Register hooks on all modules of specified types.

        Args:
            module_types: List of module types to hook
            hook_type: Type of hook ("forward" or "backward")
        """
        for name, module in self.model.named_modules():
            if any(isinstance(module, m_type) for m_type in module_types):
                if hook_type == "forward":
                    self.register_forward_hook(module, name)
                elif hook_type == "backward":
                    self.register_backward_hook(module, name)
                else:
                    raise ValueError(f"Unknown hook type: {hook_type}")

    def register_attention_hooks(self):
        """Register hooks specifically for attention layers."""
        # Common attention module types
        attention_types = [
            nn.MultiheadAttention,
        ]

        # Also check for transformer-specific attention layers
        for name, module in self.model.named_modules():
            module_class_name = module.__class__.__name__
            if any(
                attn_name in module_class_name.lower()
                for attn_name in ["attention", "multihead", "selfattention"]
            ):
                self.register_forward_hook(module, name)
                logger.info(f"Registered attention hook on {name}")

    def _create_forward_hook(self, layer_name: str) -> Callable:
        """Create a forward hook function."""

        def hook_fn(module, input_tensors, output_tensors):
            if not self._hook_enabled:
                return

            hook_output = HookOutput(
                layer_name=layer_name,
                module_type=module.__class__.__name__,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                metadata={
                    "input_shapes": [t.shape for t in input_tensors if isinstance(t, torch.Tensor)],
                    "output_shape": output_tensors.shape
                    if isinstance(output_tensors, torch.Tensor)
                    else [t.shape for t in output_tensors if isinstance(t, torch.Tensor)],
                },
            )
            self.hook_outputs[layer_name].append(hook_output)

        return hook_fn

    def _create_backward_hook(self, layer_name: str) -> Callable:
        """Create a backward hook function."""

        def hook_fn(module, grad_input, grad_output):
            if not self._hook_enabled:
                return

            # Find corresponding forward output
            if layer_name in self.hook_outputs and self.hook_outputs[layer_name]:
                # Update the last hook output with gradients
                self.hook_outputs[layer_name][-1].gradients = (
                    grad_output[0] if grad_output else None
                )
            else:
                # Create new hook output with gradient info
                hook_output = HookOutput(
                    layer_name=layer_name,
                    module_type=module.__class__.__name__,
                    gradients=grad_output[0] if grad_output else None,
                    metadata={
                        "grad_shapes": [g.shape for g in grad_output if isinstance(g, torch.Tensor)]
                    },
                )
                self.hook_outputs[layer_name].append(hook_output)

        return hook_fn

    def extract_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from hook outputs.

        Returns:
            Dictionary mapping layer names to attention weights
        """
        attention_weights = {}

        for layer_name, outputs in self.hook_outputs.items():
            if not outputs:
                continue

            # Check if this is an attention layer
            if any(keyword in layer_name.lower() for keyword in ["attention", "attn"]):
                for output in outputs:
                    if output.output_tensors is not None:
                        # Handle different attention output formats
                        if isinstance(output.output_tensors, tuple):
                            # Many attention layers return (output, attention_weights)
                            if len(output.output_tensors) > 1:
                                attn_weights = output.output_tensors[1]
                                if attn_weights is not None:
                                    attention_weights[layer_name] = attn_weights
                        else:
                            # Some layers might return attention weights directly
                            attention_weights[layer_name] = output.output_tensors

        return attention_weights

    def extract_activations(
        self, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from specified layers.

        Args:
            layer_names: List of layer names to extract from (all if None)

        Returns:
            Dictionary mapping layer names to activations
        """
        activations = {}

        if layer_names is None:
            layer_names = list(self.hook_outputs.keys())

        for layer_name in layer_names:
            if layer_name in self.hook_outputs:
                outputs = self.hook_outputs[layer_name]
                if outputs and outputs[-1].output_tensors is not None:
                    output = outputs[-1].output_tensors
                    if isinstance(output, tuple):
                        output = output[0]
                    activations[layer_name] = output

        return activations

    def extract_gradients(self, layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract gradients from specified layers.

        Args:
            layer_names: List of layer names to extract from (all if None)

        Returns:
            Dictionary mapping layer names to gradients
        """
        gradients = {}

        if layer_names is None:
            layer_names = list(self.hook_outputs.keys())

        for layer_name in layer_names:
            if layer_name in self.hook_outputs:
                outputs = self.hook_outputs[layer_name]
                if outputs and outputs[-1].gradients is not None:
                    gradients[layer_name] = outputs[-1].gradients

        return gradients

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info(f"Removed {len(self.hooks)} hooks")

    def clear_outputs(self):
        """Clear stored hook outputs."""
        self.hook_outputs.clear()

    def enable_hooks(self):
        """Enable hook execution."""
        self._hook_enabled = True

    def disable_hooks(self):
        """Disable hook execution without removing hooks."""
        self._hook_enabled = False

    def get_layer_names(self) -> List[str]:
        """Get names of all layers with registered hooks."""
        return list(self.hook_outputs.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about hook outputs."""
        stats = {
            "num_hooks": len(self.hooks),
            "num_layers": len(self.hook_outputs),
            "total_outputs": sum(len(outputs) for outputs in self.hook_outputs.values()),
            "layers": {},
        }

        for layer_name, outputs in self.hook_outputs.items():
            if outputs:
                stats["layers"][layer_name] = {
                    "num_outputs": len(outputs),
                    "has_gradients": any(o.gradients is not None for o in outputs),
                    "module_type": outputs[0].module_type if outputs else None,
                }

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.clear_hooks()
        self.clear_outputs()
