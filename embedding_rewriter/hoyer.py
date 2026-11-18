import numpy as np
import torch
import torch.nn.functional as F
from .model import HoyerModel


def hoyer_score(h1: torch.Tensor, h2: torch.Tensor, eps: float = 1e-8, dim=1) -> torch.Tensor:
    """
    Calculates the Hoyer sparsity score between two batches of embeddings.

    The score is normalized to be between 0 and 1. A higher score indicates
    greater sparsity in the difference vector between h1 and h2.

    Args:
        h1 (torch.Tensor): The first batch of embeddings.
                           Shape: (..., embedding_size)
        h2 (torch.Tensor): The second batch of embeddings.
                           Shape: (..., embedding_size)
        eps (float): A small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: A tensor of Hoyer scores.
                      Shape: (...)
    """
    # Ensure the embedding dimensions match
    if h1.shape[dim] != h2.shape[dim]:
        raise ValueError(
            f"Embedding dimensions must match. Got {h1.shape[dim]} and {h2.shape[dim]}"
        )

    embedding_size = h1.shape[dim]

    # The Hoyer function is undefined for d=1
    if embedding_size <= 1:
        raise ValueError("Hoyer sparsity is not defined for embedding_size <= 1.")

    # Calculate the difference vector
    diff = h1 - h2

    # Calculate L1 and L2 norms along the embedding dimension
    l1_norm = torch.linalg.norm(diff, ord=1, dim=dim)
    l2_norm = torch.linalg.norm(diff, ord=2, dim=dim)

    # Calculate the ratio, adding epsilon for numerical stability
    ratio = l1_norm / (l2_norm + eps)

    # Calculate the Hoyer score
    sqrt_d = np.sqrt(embedding_size)
    hoyer = (sqrt_d - ratio) / (sqrt_d - 1)

    return hoyer


def hoyer_loss(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: torch.Tensor,
    temperature: float = 0.1,
    reduction='mean',
    negative_mode='unpaired'
) -> torch.Tensor:
    """
    Computes the contrastive loss using the Hoyer sparsity function as the similarity metric.

    This loss function encourages the model to learn embeddings where the difference
    vector between a query and its positive key is sparse (high Hoyer score),
    and the difference vectors between a query and its negative keys are dense
    (low Hoyer score).

    Args:
        query (torch.Tensor): Query embeddings.
                              Shape: (batch_size, embedding_size)
        positive_key (torch.Tensor): Positive key embeddings.
                                     Shape: (batch_size, embedding_size)
        negative_keys (torch.Tensor): Negative key embeddings.
                                      Shape: (batch_size, num_negatives, embedding_size)
        temperature (float): Temperature scaling factor for the softmax function.

    Returns:
        torch.Tensor: The computed scalar loss value.
    """
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query = F.normalize(query, dim=-1)
    positive_key = F.normalize(positive_key, dim=-1)
    if negative_keys is not None:
        negative_keys = F.normalize(negative_keys, dim=-1)

    # 1. Calculate the Hoyer score for the (query, positive_key) pairs.
    # This acts as our "positive similarity".
    # Shape: (batch_size,)
    positive_scores = hoyer_score(query, positive_key, dim=-1)

    # 2. Calculate the Hoyer scores for the (query, negative_keys) pairs.
    if negative_mode == 'unpaired':
        # Cosine between all query-negative combinations
        negative_scores = hoyer_score(query, negative_keys, dim=-1)

    elif negative_mode == 'paired':
        # We need to expand the query to match the shape of the negative keys.
        # Reshape query from (batch_size, embedding_size) to
        # (batch_size, 1, embedding_size) and then expand to
        # (batch_size, num_negatives, embedding_size)
        expanded_query = query.unsqueeze(1).expand(-1, negative_keys.shape[-2], -1)
        negative_scores = hoyer_score(expanded_query, negative_keys)

    # 3. Combine positive and negative scores for the InfoNCE loss calculation.
    # We want the first logit to correspond to the positive pair.
    # Reshape positive_scores to (batch_size, 1) for concatenation.
    # Final shape: (batch_size, 1 + num_negatives)
    all_scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)

    # 4. Apply the temperature scaling.
    logits = all_scores / temperature

    # 5. The labels are the indices of the positive pairs. Since we arranged
    # our scores so the positive is always at index 0, the labels are all zeros.
    labels = torch.zeros(len(all_scores), dtype=torch.long, device=query.device)

    # 6. Compute the cross-entropy loss (which includes the softmax operation).
    loss = F.cross_entropy(logits, labels, reduction=reduction)

    return loss
