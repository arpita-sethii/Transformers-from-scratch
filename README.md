# Transformer Encoder From Scratch in NumPy (With Full Backpropagation)

This project implements a **Transformer Encoder from scratch using pure NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The objective is to deeply understand the **mathematics and gradient flow** behind Transformers by manually implementing both the **forward pass and the complete backward pass**.

> No autograd. No deep learning libraries. Everything is computed explicitly.

---

## 🚀 Key Features

- Learnable token embedding matrix
- Sinusoidal positional encoding
- Scaled dot-product self-attention
- Query, Key, Value (QKV) projections
- Residual connections
- Layer Normalization (forward + backward)
- Position-wise Feed Forward Network (FFN)
- Output projection to vocabulary space
- Softmax + Cross-Entropy loss
- **Full backpropagation implemented manually**
- Gradient flow verified across all components

---

## 🧠 Model Architecture

```
Input Tokens
   ↓
Token Embedding + Positional Encoding
   ↓
Self-Attention (Q, K, V)
   ↓
Residual Connection + LayerNorm
   ↓
Feed Forward Network (ReLU)
   ↓
Residual Connection + LayerNorm
   ↓
Linear Projection → Vocabulary
   ↓
Softmax + Cross-Entropy Loss
```

---

## 📊 Training Setup

- Dataset: Synthetic sequence prediction task  
  (Each token learns to predict the next token in the sequence)
- Vocabulary size: 20
- Sequence length: 4
- Embedding dimension (`d_model`): 32
- Feed-forward hidden dimension: 64
- Optimizer: Vanilla SGD
- Loss: Categorical Cross-Entropy

The training loss decreases steadily over epochs, confirming that gradients are correctly flowing through the entire Transformer block.

---

## 🔁 Backpropagation Coverage

The backward pass is manually derived and implemented for:

- Vocabulary projection layer
- Feed-forward network (both linear layers)
- ReLU activation
- Layer normalization (mean and variance gradients)
- Residual connections
- Attention output projection
- Scaled dot-product attention
- Softmax (both output and attention)
- Query, Key, Value projection matrices
- Input embeddings

This ensures **end-to-end differentiability without automatic differentiation**.

---

## 📈 Results

- Training loss decreases monotonically across epochs
- Gradients remain stable (no exploding or vanishing behavior)
- Confirms correctness of attention and LayerNorm backpropagation

---

## 🛠 Technologies Used

- Python
- NumPy
- Matplotlib (for debugging and gradient visualization)

---

## 🎯 Learning Outcomes

By completing this project, you gain a deep understanding of:

- How Transformers work internally
- Why residual connections and normalization are critical
- How attention gradients propagate
- The exact math behind modern Transformer architectures

This project is intended for **educational and research purposes**.

---

## 📌 Notes

- This implementation focuses on **clarity and correctness**, not speed
- Only a single Transformer encoder block is implemented
- Multi-head attention and stacking multiple layers are left as extensions

---

## 📄 License

This project is open-source and free to use for learning and research.
