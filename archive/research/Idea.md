# Masked Attractor Discovery

*Finding “undiscovered” concepts by learning the gaps between known points in embedding space*

## TL;DR

Treat knowledge as points in embedding space; treat **missing** knowledge as **attractors** residing in the **gaps**.

1. Train/obtain a **teacher** embedding on full text.
2. **Mask** a held-out concept set in a **student** corpus.
3. Train a small **student projector** to reconstruct the teacher vector of the missing token **from its context**.
4. Decode by **nearest neighbors** to reveal the hidden concept, and use **probe vectors** (e.g., `cat + bicycle`) to snap toward “gap” attractors.
   Works with tiny models and scales to transformers and domain corpora.

---

## Why this matters

Distillation and embedding learning compress relational structure into vectors. If knowledge is a set of points, **the interesting part is often the relational fabric in between**. By forcing a learner to infer a masked vector from context **without ever seeing its symbol**, we operationalize “knowledge in the gaps.” This is useful for:

* **Hypothesis surfacing** in research corpora
* **Ontology completion** and schema suggestion
* **Accelerated literature mapping** (what should exist, given what does)
* **Product/discovery prompts** (“Cool, but have you tried *Cats on Bicycles*?”)

---

## Core idea (one paragraph)

Build a teacher embedding $E_t$. Hide a concept set $H$ (mask all occurrences) when training a student $f_\theta$ to map context windows to vectors. Supervise on the **teacher vector** of the missing token. At inference, use $f_\theta$ to project contexts or synthetic **probes** into teacher space; then decode via nearest neighbors or cluster centroids. Stable convergences indicate **attractors**—coherent, often novel conceptual neighborhoods.

---

## Minimal method (works today, tiny compute)

1. **Teacher embedding**: Build with PPMI+SVD or Word2Vec/fastText on the **full** text.
2. **Masking**: Choose held-out concepts $H$ and replace them with `[MASK]` in a **student** copy of the corpus.
3. **Student projector**: A tiny model (even linear) maps a context embedding $x$ → predicted vector $\hat{v}$.
4. **Loss**: cosine regression toward the teacher vector $v=E_t(h)$ of the true masked token $h$.

   ```python
   loss = 1 - cos(v_hat, v) + λ * ||v_hat||^2
   ```
5. **Decoding & discovery**: Nearest neighbors in teacher space label $\hat{v}$. Cluster many $\hat{v}$ to surface stable attractors.
6. **Probe mode**: Feed synthetic mixtures (e.g., `emb(cat) + emb(bicycle)`) through the projector to “snap” into a gap attractor and decode.

---

## Notebook (ready to run)

I’ve packaged a self-contained notebook (no internet needed) that implements this pipeline on a toy corpus (including **Cats on Bicycles** examples), trains a teacher (PPMI+SVD), masks concepts, learns the projector, evaluates Top-k recovery, and demonstrates **probe-based** discovery:

**[Download: Masked\_Attractor\_Discovery.ipynb](sandbox:/mnt/data/Masked_Attractor_Discovery.ipynb)**

What it shows:

* Top-1 recovery of masked tokens on the toy set
* Probes like `["cat","bicycle"]` snapping to the right region
* A clean scaffold you can swap to your own corpora

---

## “Good enough” pseudocode

```python
# Build teacher embedding E_t on full corpus
E_t = train_embedding(corpus_full)

# Mask concept set H in a student corpus
student_corpus = replace_tokens(corpus_full, H, "[MASK]")

# Build (context -> target vector) dataset
X = [context_embed(window_around_mask, E_t)]
Y = [E_t[true_missing_token]]

# Train tiny projector (linear or mini-transformer)
theta = fit_projector(X, Y, loss = 1 - cos(Wx, y) + λ||Wx||²)

# Discovery
v_hat = projector(context_or_probe)
label = nearest_neighbors(E_t, v_hat)
```

---

## Interpreting attractors

* **Nearest neighbors**: The top-k tokens around $\hat{v}$ provide names.
* **Cluster exemplars**: For many contexts/probes, cluster $\hat{v}$ and inspect exemplars.
* **Cross-model triangulation**: Agreement across embeddings/models increases confidence.

---

## Simple evaluation

* **Recoverability**: Top-k accuracy on masked concepts.
* **Vector fidelity**: mean cosine$(\hat{v}, v)$.
* **Stability**: Repeat with different seeds/splits/models.
* **Novelty**: How often are attractors dominated by mixed-domain neighbors vs trivial synonyms?

---

## Scaling up (recommended next)

* **Contrastive head** (InfoNCE): score $s(\text{ctx}, w)=\cos(f_\theta(\text{ctx}),E_t(w))$; train with positives+negatives for sharper retrieval.
* **Mini-Transformer** student: replace the linear projector with a tiny transformer over context tokens.
* **Domain corpora**: Finance, biomed, codebases, your Substrate docs—mask domain entities to surface implied concepts.
* **Autoencoder bridge**: Train a light decoder from embedding space to token distributions to improve “naming” of novel attractors.
* **Active discovery**: Generate probes by analogy directions and gradient steps; prioritize high-uncertainty/high-density regions.

---

## Applications

* **Research scouting**: Given recent papers, which *implied but unwritten* links cluster nearby?
* **Ontology growth**: Suggest new nodes/relations with semantic evidence.
* **Product ideation**: Probe combinations (“cats”+“bicycles”) to surface viable unexplored seams.
* **Editorial tooling**: Feed a topic bundle, get back attractor labels and 1-paragraph capsule summaries for each.

---

## Limitations & pitfalls

* **Labeling is approximate**: NN labels are proxies; consider human-in-the-loop curation.
* **Embedding biases**: Teacher embeddings inherit corpus biases; triangulate across corpora/models.
* **Name vs concept**: A strong attractor may resist neat naming—treat the vector neighborhood (and examples) as the real object.

---

## Ethical note

Discovery tooling can accelerate good ideas—and bad ones. Curate corpora, add safety filters, and build review steps for sensitive domains.

---

## Reproduce it on your data (5 steps)

1. Replace the toy corpus with your text (one sentence per line is fine).
2. Keep the **teacher** step (PPMI+SVD or Word2Vec).
3. Define a held-out concept set $H$ to mask.
4. Train the projector; verify Top-k recovery.
5. Run many **probes**; cluster outputs; review attractors.

---

## Example: “Cats on Bicycles”

* Mask `{bicycle, helmet, handlebars, …}`; train projector.
* Probe with `["cat","bicycle"]` or contexts about pedaling and balance.
* You’ll see convergences around the *cycling-feline* neighborhood (helmet/ears/bell/handlebars).
* That neighborhood is your **attractor**—the gap prediction the system discovered without the explicit symbol present.

---

## Closing

This is a small, sturdy lever: **learn the gaps, not just the points**. It plays nicely with distillation, scales up to transformers, and is fun to drive with probes. If you publish or remix this, I’d love to see the domain-specific attractors you uncover.

—C & GPT-5 Thinking
