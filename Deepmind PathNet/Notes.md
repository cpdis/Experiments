---
Title: PathNet: Evolution Channels Gradient Descent in Super Neural Networks
Author(s): 
Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha†, Andrei A. Rusu, Alexander Pritzel, Daan Wierstra; Google DeepMind
Keywords: giant networks, path evolution algorithm, evolution and learning, continual learning, transfer learning, multitask learning, basal ganglia
Start: 6/28/17
End:  
---

## Summary
PathNet is a new deep learning architecture that combines modular deep learning, meta-learning, and reinforcement learning. From the paper:

>For artificial general intelligence (AGI) it would be efficient if multiple users trained the same giant neural network, permitting parameter reuse, without catastrophic forgetting. PathNet is a first step in this direction. It is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks.

---
## Notes
- Reuses a network consisting of many networks on multiple tasks
    - Layers of neural networks interconnected through different search methods
- Includes aspects of transfer learning, continual learning, and multitask learning. This allows the network to (in theory) continuously adapt.
- Conditional Logic
    - Programming
        - Computation
        - Conditional logic
        - Iteration/recursion
    - Neuron
        - Computational unit (sum of products)
        - Conditional unit (activation function)
        - Layers added which led to deep learning
    - Add loops into network
        - Led to RNNs
        - RNNs are chaotic so memory (buffer) is needed with led to LSTM
---
## Resources
- https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273

- https://medium.com/intuitionmachine/is-conditional-logic-the-new-deep-learning-hotness-96832774907b