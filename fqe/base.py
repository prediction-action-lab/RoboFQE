"""Base classes shared by both FQE variants (`fqe.scratch`, `fqe.pretrained`).

Ported from (merged into one module for convenience):
  - opebench/algorithms/base.py
  - opebench/policies/base.py
  - opebench/utils/common.py
of RoboOPE main commit 6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>).
"""
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.distributions as D


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BaseOPEAlgorithm(ABC):
    """Base class for all OPE algorithms."""

    def __init__(self):
        set_seed(42)

    @abstractmethod
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        initial_batch: Dict[str, torch.Tensor],
        policy,
    ) -> Dict[str, float]:
        """Single update step.

        Args:
            batch: Batch from dataset.
            initial_batch: Batch of initial states.
            policy: PolicyWrapper.

        Returns:
            info: Dictionary of losses/metrics.
        """
        pass

    @abstractmethod
    def estimate(self, dataloader, policy) -> float:
        """Estimate policy value.

        Args:
            dataloader: DataLoader.
            policy: PolicyWrapper.

        Returns:
            estimated_value: Estimated policy value.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model checkpoints."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model checkpoints."""
        pass


class PolicyWrapper(ABC):
    """Unified interface for different RL libraries."""

    def __init__(self):
        set_seed(42)

    @abstractmethod
    def get_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample action from policy.

        Args:
            obs: Dict of observations.

        Returns:
            actions: (B, action_dim).
        """
        pass

    def get_action_distribution(self, obs: Dict[str, torch.Tensor]) -> D.Distribution:
        """Get action distribution (for stochastic policies)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support action distributions. "
            "This method is only available for stochastic policies (e.g., Gaussian)."
        )

    def get_action_batch(
        self, obs: Dict[str, torch.Tensor], n_samples: int = 1
    ) -> torch.Tensor:
        """Sample multiple actions (optional, for continuous actions).

        Default: call get_action n_samples times.
        """
        actions = torch.stack([self.get_action(obs) for _ in range(n_samples)], dim=1)
        return actions

    def get_log_prob(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log π(a|s)."""
        dist = self.get_action_distribution(obs)
        log_prob = dist.log_prob(actions)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob
