# NanoGPT — Character-Level Transformer from Scratch

A decoder-only transformer (à la Karpathy's nanoGPT) implemented **from first principles in PyTorch** — no `nn.Transformer`, no library attention — plus a controlled experiment on a real architectural hyperparameter.

## Problem

To genuinely understand how modern LLMs work, I built a character-level decoder-only transformer from the ground up rather than calling a high-level API. The goal was twofold: implement the **full attention mechanism** — Q/K/V projections, scaled dot-product attention, causal masking, multi-head attention, residual + layer-norm blocks — and then run a **controlled study** on the number of attention heads to see how it affects learning, all on CPU within a sensible training budget.

## Approach

- **Character-level tokenisation & data pipeline** on the tiny-Shakespeare corpus (65 unique characters as the vocabulary): char↔int encode/decode, 64-character context windows, train/validation split.
- **Self-attention from first principles.** Each head projects the input to Q, K, V via bias-free linear layers, computes `softmax(QKᵀ/√d_h)·V`, and applies a lower-triangular **causal mask** so each position only attends to the past. The `√d_h` scaling is documented (stops dot products from saturating the softmax).
- **Multi-head attention + transformer block.** `n_head` heads run in parallel over subspaces, concatenated and projected back. Stacked 4 blocks, each = self-attention + feed-forward (4× expansion 64→256→64, ReLU), wrapped with **pre-norm and residual connections**.
- **Training, loss & sampling.** Final linear layer maps each 64-dim output to 65 character logits; cross-entropy loss; AdamW (`lr=1e-3`); 5,000 steps with periodic validation. Generation uses **temperature** scaling + `torch.multinomial` sampling.
- **Controlled hyperparameter study.** Held everything fixed (`n_embd=64`, `n_layer=4`, `block_size=64`, 5,000 steps) and compared **Config A (`n_head=2`, `d_h=32`)** vs. **Config B (`n_head=4`, `d_h=16`)**, tracking train/val loss curves.
- **Quantitative text analysis** — instead of eyeballing output, measured % of lines starting with a capital, punctuation rate, correctly-formatted character names, and the effect of temperature on coherence vs. variety.

## Results

- Both configs dropped steeply in the first ~1,000 steps (loss ~4.3 → ~1.7); val loss tracked train loss closely → **no obvious overfitting**.
- **Config B (4 heads)** was marginally better: **val loss 1.7435 vs. 1.7707** for 2 heads — interpreted correctly (with only 64 total embedding dims there's limited room for heads to specialise, though more subspaces give slightly more flexibility).
- The **212K-parameter** model picked up real surface structure: ~90% of lines start with a capital, character names (ROMEO, WARWICK) appear in correct all-caps, punctuation ~4.1% in sensible positions, archaic words ("thy", "dost") emerge. Limitation analysed: invented words and no real grammar — expected for this model size and 64-char context. Lower τ (0.7) → repetitive but coherent; higher τ (1.3) → varied but less readable.

## Tech stack

Python · PyTorch (`nn.Module`, custom attention) · NumPy · Matplotlib

## Data

tiny-Shakespeare corpus — **downloaded automatically** by the notebook on first run (no manual setup needed).

## Run

```bash
pip install -r requirements.txt
jupyter notebook nanogpt_transformer.ipynb
```
