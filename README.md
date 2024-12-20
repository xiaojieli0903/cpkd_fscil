# Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning

🎉🎉🎉 In this work, we propose Continuous Knowledge-Preserving Decomposition for Few-shot Class-incremental Learning (CKPD-FSCIL), a framework that efficiently decomposes model’s weights into two complementary parts: one that compacts existing knowledge (knowledge-sensitive components) and another carries redundant capacity to accommodate new abilities (redundant-capacity components). The decomposition is guided by a covariance matrix from replay samples such that the decomposed principal components align closely with the classification abilities of these representative samples. During adaptation, we freeze the knowledge-sensitive components and only adapt the redundant-capacity components, fostering plasticity for new abilities while minimizing interference with existing knowledge, without changing model architecture or increasing inference overhead. Additionally, CKPD introduces an adaptive layer selection strategy to identify layers with the most redundant capacity, dynamically allocating adapters across layers.

Work in progress 🚀🚀🚀
- [ ] Code and Instructions
- [ ] Checkpoints
- [ ] ArXiv release

## ✏️ Citation
```
```
## 👍 Acknowledgments
This codebase builds on [FSCIL](https://github.com/NeuralCollapseApplications/FSCIL) and [Mamba-FSCIL](https://github.com/xiaojieli0903/Mamba-FSCIL). Thanks to all the contributors.