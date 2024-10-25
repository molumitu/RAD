import numpy as np
import torch
from typing import Iterable, List, Optional, Tuple

def cal_bpograd(grad_list, bpo_coefficient) -> None:
    num_grads = len(grad_list)
    grad_vec = torch.cat(
        list(
            map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad_list)
        ),
        dim=0,
    )

    regularized_grad = bpograd(grad_vec, num_grads, bpo_coefficient)
    return regularized_grad


def bpograd(gradient_vector, num_grads, bpo_coefficient=0.5):
    """
    Optimize gradients based on given tasks and BPO coefficient.

    Args:
        gradient_vector (torch.Tensor): A tensor of shape [num_grads, dim].
        num_grads (int): Number of tasks.
        bpo_coefficient (float): Coefficient for BPO.

    Returns:
        torch.Tensor: Optimized gradient.
    """
    gradients = gradient_vector
    covariance_matrix = gradients.mm(gradients.T).cpu()
    
    # scale = (torch.diag(covariance_matrix) + 1e-4).sqrt().mean()
    normalized_covariance = covariance_matrix 
    # / scale.pow(2)
    
    mean_covariance = normalized_covariance.mean(1, keepdims=True)
    overall_mean = mean_covariance.sum(0, keepdims=True)

    weights = torch.zeros(num_grads, 1, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=1)

    adjustment_factor = (overall_mean + 1e-4).sqrt() * bpo_coefficient

    best_weights = None
    best_objective = np.inf
    
    for iteration in range(21):
        optimizer.zero_grad()
        softmax_weights = torch.softmax(weights, 0)
        objective = (softmax_weights.T.mm(mean_covariance) + 
                     adjustment_factor * (softmax_weights.T.mm(normalized_covariance).mm(softmax_weights) + 1e-4).sqrt())

        if objective.item() < best_objective:
            best_objective = objective.item()
            best_weights = weights.clone()
        
        if iteration < 20:
            objective.backward()
            optimizer.step()

    softmax_best_weights = torch.softmax(best_weights, 0)
    print('softmax_best_weights', softmax_best_weights)
    gradient_norm = (softmax_best_weights.T.mm(normalized_covariance).mm(softmax_best_weights) + 1e-4).sqrt()

    lambda_param = adjustment_factor.view(-1) / (gradient_norm + 1e-4)
    optimized_gradient = ((1 / num_grads + softmax_best_weights * lambda_param).view(-1, 1).to(gradients.device) * gradients).sum(0)
    
    weight_1 = (1 / num_grads + softmax_best_weights * lambda_param).view(-1, 1)[0]
    weight_2 = (1 / num_grads + softmax_best_weights * lambda_param).view(-1, 1)[1]
    weight_1, weight_2 = float(weight_1.detach().cpu())/(1+bpo_coefficient), float(weight_2.detach().cpu())/(1+bpo_coefficient)
    return optimized_gradient, weight_1, weight_2
    
def apply_grad_vector_to_params(
    model_params: Iterable[torch.Tensor], grad_vector: torch.Tensor, accumulate: bool = False
):
    """Apply gradient vector to model parameters.

    Args:
        model_params (Iterable[torch.Tensor]): Iterable of model parameter tensors.
        grad_vector (torch.Tensor): A single vector representing the gradients.
        accumulate (bool): Whether to accumulate the gradients or overwrite them.
    """
    # Ensure grad_vector is of type Tensor
    if not isinstance(grad_vector, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got: {type(grad_vector).__name__}")

    # Pointer for slicing the gradient vector for each parameter
    pointer = 0
    for param in model_params:
        num_elements = param.numel()
        # Slice the vector and reshape it to match the parameter's shape
        if accumulate:
            param.grad = (param.grad + grad_vector[pointer:pointer + num_elements].view_as(param).data)
        else:
            param.grad = grad_vector[pointer:pointer + num_elements].view_as(param).data

        # Update the pointer
        pointer += num_elements