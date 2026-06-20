# Notices

This repository contains original additions for temporal reward shaping in VLM-guided reinforcement learning and code adapted from third-party projects.

## FuRL

The training structure and FuRL-style baseline code are derived from:

- Yuwei Fu et al. "FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning". Proceedings of the 41st International Conference on Machine Learning, 2024.
- Upstream repository: https://github.com/fuyw/FuRL

At the time this notice was prepared, the upstream FuRL repository did not show a license file in its GitHub repository listing. This repository keeps provenance explicit so users can review upstream terms before reuse or redistribution.

## CLIP

The `Code/clip/` directory vendors code and tokenizer assets from OpenAI CLIP. Its MIT license is preserved in `Code/clip/LICENSE`:

- https://github.com/openai/CLIP

## LIV and Other Dependencies

The code adapts LIV components and loads LIV checkpoints. The LIV MIT license is preserved in `Code/LIV_LICENSE`. It also uses dependencies including Meta-World, Gymnasium, JAX, Flax, PyTorch, W&B, and related scientific Python packages. These dependencies are not relicensed by this repository.

## Repository License

The `LICENSE` file applies to this repository's original additions unless a file or third-party component states otherwise.
