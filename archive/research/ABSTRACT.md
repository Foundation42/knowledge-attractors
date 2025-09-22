# Abstract

**Masked Attractor Discovery** learns the gaps between known points in embedding space. We train a teacher embedding on full text, mask a concept set, and teach a small student to reconstruct the missing vectors from context. Probes then snap into high-density "gap" regions—i.e., implied concepts. On toy and domain corpora we achieve 88–93% Top-1 masked recovery (100% Top-5, mean cosine ≥ 0.92), surface stable attractors (84% persistence), and drive two applications: a **Creativity Add-On** that emits grounded idea cards, and **RLAD**, a curiosity loop where the model chooses what to learn next by geometric information gain.

## Key Innovation

Traditional approaches learn the points (known concepts). We learn the **gaps** (undiscovered concepts). This transforms:

- **Creativity**: From random generation → guided exploration of semantic gaps
- **Learning**: From passive absorption → active curiosity-driven discovery
- **Discovery**: From searching what exists → finding what should exist

## Figure Caption

**Figure 1**: Masked Attractor Discovery. *Left*: Teacher manifold (PPMI+SVD) with masked concepts highlighted in red. *Right*: Discovered attractors (red stars) from masked contexts and probe attractors (colored triangles). Probe combinations like `cat+bicycle` reliably land in semantically coherent regions that correspond to the masked concepts, demonstrating that knowledge lives in the gaps between points.

## One-Line Summary

> Learn the gaps, not just the points—knowledge lives in the spaces between.