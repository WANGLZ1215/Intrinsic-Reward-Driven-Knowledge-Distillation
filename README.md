# Intrinsic Reward-Driven Knowledge Distillation (IRKD)

Official implementation of the Master’s thesis:
“Intrinsic Reward-Driven Knowledge Distillation for Mathematical Reasoning in Large Language Models”.

## Overview

This project introduces IRKD, a reinforcement-learning-based knowledge distillation framework that reconstructs token-level intrinsic rewards directly from teacher logits via an inverse soft Bellman formulation.

Unlike conventional distillation methods that match output distributions, IRKD extracts the latent value structure embedded in the teacher model and transforms it into a dense reward signal. The student model is optimized using PPO with KL regularization under a LoRA-based parameter-efficient setting.

## Key Contributions

- Interpreting teacher logits as soft Q-functions under MaxEnt RL
- Deriving intrinsic rewards via inverse soft Bellman operator
- Intrinsic-only policy optimization without external reward models
- Empirical validation on GSM8K using Qwen2.5-32B → Qwen2.5-7B

