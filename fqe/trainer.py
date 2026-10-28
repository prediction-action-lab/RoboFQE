"""OPETrainer — generic train loop for OPE algorithms.

Ported verbatim from opebench/trainer/ope_trainer.py of RoboOPE main commit
6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>). Only the import of
set_seed was rewritten to the local relative path.
"""
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import set_seed


class OPETrainer:
    """
    Trainer for OPE algorithms

    Handles:
    - DataLoader creation
    - Training loop
    - Logging (wandb)
    - Checkpointing
    - Periodic evaluation
    """

    def __init__(
        self,
        algorithm,
        dataset,
        policy,
        behavior_policy=None,
        seed: int = 42,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None,
    ):
        """
        Args:
            algorithm: BaseOPEAlgorithm instance
            dataset: OPEDataset instance
            policy: Target policy (want to estimate); PolicyWrapper instance
            batch_size: Batch size for training
            device: Device to use
            log_interval: Log every N iterations
            eval_interval: Evaluate every N iterations
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use wandb logging
            wandb_config: Config dict for wandb.init()
        """
        set_seed(seed)
        self.algorithm = algorithm
        self.dataset = dataset
        self.policy = policy
        self.behavior_policy = behavior_policy
        self.seed = seed
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.use_wandb = use_wandb

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Initialize wandb
        if self.use_wandb:
            import wandb
            wandb.init(**(wandb_config or {}))

        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            import random
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Generator for DataLoader
        g = torch.Generator()
        g.manual_seed(seed)

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        g_eval = torch.Generator()
        g_eval.manual_seed(seed)

        self.eval_loader = DataLoader(
            dataset,
            batch_size=4096,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g_eval,
        )

        self.initial_states = self.dataset.get_initial_states()

    def train(self, num_iterations: int):
        """
        Training loop

        Args:
            num_iterations: Number of update iterations
        """
        print(f"Starting training for {num_iterations} iterations...")

        iteration = 0
        epoch = 0
        pbar = tqdm(total=num_iterations, desc="Training")

        while iteration < num_iterations:
            # Start of new epoch
            for batch in self.train_loader:
                if iteration >= num_iterations:
                    break

                actual_batch_size = batch["actions"].shape[0]

                # Sample initial states
                num_initial = len(self.initial_states[list(self.initial_states.keys())[0]])
                initial_indices = np.random.choice(
                    num_initial,
                    actual_batch_size,
                    replace=True
                )
                initial_batch = {
                    'obs': {k: v[initial_indices] for k, v in self.initial_states.items()}
                }

                # Update
                info = self.algorithm.update(batch=batch, initial_batch=initial_batch, policy=self.policy)

                # Add epoch info
                info['epoch'] = epoch

                # Log to console
                if iteration % self.log_interval == 0:
                    log_str = f"Iter {iteration} (Epoch {epoch}): " + ", ".join(
                        [f"{k}={v:.4f}" for k, v in info.items() if k != 'epoch'])
                    pbar.set_postfix_str(log_str)

                # Log to wandb
                if self.use_wandb:
                    import wandb
                    wandb.log({f"train/{k}": v for k, v in info.items()}, step=iteration)

                # Evaluate periodically
                if iteration % self.eval_interval == 0 and iteration > 0:
                    eval_results = self._evaluate_internal()

                    # Print evaluation results
                    print(f"\nIteration {iteration} (Epoch {epoch}):")
                    for key, value in eval_results.items():
                        print(f"  {key}: {value:.4f}")

                    if self.use_wandb:
                        wandb.log({
                            **{f"eval/{k}": v for k, v in eval_results.items()},
                            "eval/epoch": epoch
                        }, step=iteration)

                # Save checkpoint
                if self.save_dir and iteration % 10000 == 0:
                    save_path = os.path.join(self.save_dir, f"ckpt_iter_{iteration}_epoch_{epoch}.pt")
                    self.algorithm.save(save_path)

                iteration += 1
                pbar.update(1)

            # End of epoch
            epoch += 1
            print(f"\nCompleted epoch {epoch}")

        pbar.close()

        # Final checkpoint
        if self.save_dir:
            final_path = os.path.join(self.save_dir, f"final_iter_{iteration}_epoch_{epoch}.pt")
            self.algorithm.save(final_path)
            print(f"Saved final checkpoint to {final_path} (Iteration: {iteration}, Epoch: {epoch})")

        # Final evaluation
        final_results = self.evaluate()

        if self.use_wandb:
            import wandb
            wandb.log({
                **{f"eval/final_{k}": v for k, v in final_results.items()},
                "eval/final_epoch": epoch
            }, step=num_iterations)

        print(f"\nTraining completed: {iteration} iterations, {epoch} epochs")

    def _evaluate_internal(self) -> Dict[str, float]:
        """Internal evaluation (without printing header)."""
        if self.behavior_policy is not None:
            result = self.algorithm.estimate(self.eval_loader, self.policy, behavior_policy=self.behavior_policy)
        else:
            result = self.algorithm.estimate(self.eval_loader, self.policy)

        # Handle both dict and float returns for backward compatibility
        if isinstance(result, dict):
            return result
        else:
            return {"policy_value": result}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate policy value using trained algorithm."""
        print("Evaluating policy value...")
        eval_results = self._evaluate_internal()

        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")

        return eval_results

    def finish(self):
        """Finish wandb logging."""
        if self.use_wandb:
            import wandb
            wandb.finish()
