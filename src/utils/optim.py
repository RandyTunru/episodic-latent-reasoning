from torch import nn
from typing import List, Tuple, Dict, Any, Optional

def param_groups_with_decay(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    no_decay_patterns: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """Separate parameters into groups with and without weight decay.

    Args:
        model: The model containing parameters to group.
        weight_decay: The weight decay value to apply to applicable parameters.
        lr: The learning rate to apply to all parameter groups.
        betas: The beta values for the optimizer.
        no_decay_patterns: Optional tuple of parameter name patterns to exclude from weight decay.

    Returns:
        A list of parameter groups, each represented as a dictionary with keys
        'params' and 'weight_decay'.
    """
    if no_decay_patterns is None:
        # Exclude normalization layers from weight decay (matches by parameter name, not layer type).
        # This would catch any layer with "norm" in its name, eg. "norm1", "layer_norm".
        # Change this to whatever you name your normalization layers if you use a different naming convention.
        no_decay_patterns = ("norm",)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        if len(param.shape) == 1 or name.endswith(".bias") or any(pat in name for pat in no_decay_patterns):
            # By default, parameters with 1D shape and biases are excluded from weight decay
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr, "betas": betas},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr, "betas": betas},
    ]