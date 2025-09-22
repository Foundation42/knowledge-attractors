#!/usr/bin/env python3
"""
Advanced Attractor Mining with Batch Discovery and Auto-Description
"""

import numpy as np
from collections import defaultdict, Counter
import hdbscan
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from masked_attractor import MaskedAttractorTrainer
import warnings
warnings.filterwarnings('ignore')


class AttractorMiner:
    """Batch mining and analysis of semantic attractors"""

    def __init__(self, trainer: MaskedAttractorTrainer):
        self.trainer = trainer
        self.discovery = trainer.discovery
        self.teacher = trainer.teacher
        self.mined_attractors = []
        self.clusters = []
        self.cluster_descriptions = []

    def batch_mine(self, n_probes: int = 5000, strategies: List[str] = None) -> List[Dict]:
        """Mine attractors using multiple probe strategies"""

        if strategies is None:
            strategies = ['random', 'analogy', 'interpolation', 'perturbation']

        print(f"Mining {n_probes} probes across {len(strategies)} strategies...")
        attractors = []

        probes_per_strategy = n_probes // len(strategies)

        for strategy in strategies:
            print(f"  {strategy}: ", end="", flush=True)

            for i in range(probes_per_strategy):
                if i % 100 == 0:
                    print(".", end="", flush=True)

                probe_vec = self._generate_probe(strategy)
                if probe_vec is None:
                    continue

                # Apply mean-shift to find attractor
                refined_vec, neighbors = self.discovery.mean_shift(
                    probe_vec, k=30, step_size=0.5, max_iter=20
                )

                # Compute quality metrics
                coherence = self._compute_coherence(neighbors)
                novelty = self._compute_novelty(neighbors)
                density = self._compute_density(refined_vec)

                attractors.append({
                    'vector': refined_vec,
                    'neighbors': neighbors[:10],
                    'coherence': coherence,
                    'novelty': novelty,
                    'density': density,
                    'strategy': strategy
                })
            print(" done")

        self.mined_attractors = attractors
        return attractors

    def _generate_probe(self, strategy: str) -> Optional[np.ndarray]:
        """Generate probe vector using specified strategy"""

        vocab = list(self.teacher.tok2id.keys())
        d = self.teacher.embed_dim

        if strategy == 'random':
            # Pure random direction
            vec = np.random.randn(d)

        elif strategy == 'analogy':
            # Analogy-based: A - B + C
            if len(vocab) < 3:
                return None
            words = np.random.choice(vocab, size=3, replace=False)
            vecs = [self.teacher.get_embedding(w) for w in words]
            vecs = [v for v in vecs if v is not None]
            if len(vecs) < 3:
                return None
            vec = vecs[0] - vecs[1] + vecs[2]

        elif strategy == 'interpolation':
            # Linear interpolation between concepts
            if len(vocab) < 2:
                return None
            words = np.random.choice(vocab, size=2, replace=False)
            v1 = self.teacher.get_embedding(words[0])
            v2 = self.teacher.get_embedding(words[1])
            if v1 is None or v2 is None:
                return None
            alpha = np.random.random()
            vec = alpha * v1 + (1 - alpha) * v2

        elif strategy == 'perturbation':
            # Small perturbation of existing concept
            word = np.random.choice(vocab)
            base = self.teacher.get_embedding(word)
            if base is None:
                return None
            noise = np.random.randn(d) * 0.3
            vec = base + noise

        else:
            return None

        # Normalize
        return vec / (np.linalg.norm(vec) + 1e-8)

    def cluster_attractors(self, min_cluster_size: int = 10) -> Dict:
        """Cluster discovered attractors using HDBSCAN"""

        if len(self.mined_attractors) < min_cluster_size * 2:
            print("Not enough attractors for clustering")
            return {}

        print(f"\nClustering {len(self.mined_attractors)} attractors...")

        # Stack vectors
        X = np.array([a['vector'] for a in self.mined_attractors])

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',  # Works better than cosine for normalized vectors
            cluster_selection_epsilon=0.1
        )

        labels = clusterer.fit_predict(X)

        # Analyze clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"Found {n_clusters} clusters ({n_noise} noise points)")

        # Group attractors by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Compute cluster quality
        cluster_info = {}
        for label, indices in clusters.items():
            if label == -1:  # Skip noise
                continue

            cluster_vecs = X[indices]
            centroid = cluster_vecs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Get consensus neighbors
            all_neighbors = []
            for idx in indices[:20]:  # Sample for efficiency
                neighbors = self.mined_attractors[idx]['neighbors']
                all_neighbors.extend([w for w, _ in neighbors[:5]])

            # Find most common neighbors
            neighbor_counts = Counter(all_neighbors)
            consensus = neighbor_counts.most_common(10)

            cluster_info[label] = {
                'size': len(indices),
                'centroid': centroid,
                'consensus_neighbors': consensus,
                'mean_coherence': np.mean([self.mined_attractors[i]['coherence'] for i in indices]),
                'mean_density': np.mean([self.mined_attractors[i]['density'] for i in indices]),
                'indices': indices
            }

        self.clusters = cluster_info
        return cluster_info

    def generate_descriptions(self, corpus: List[List[str]] = None) -> List[Dict]:
        """Generate auto-descriptions for each cluster"""

        if not self.clusters:
            print("No clusters found. Run cluster_attractors first.")
            return []

        descriptions = []

        for label, info in self.clusters.items():
            # Get top consensus words
            top_words = [word for word, count in info['consensus_neighbors'][:5]]

            # Find nearest sentences from corpus (if provided)
            context_snippets = []
            if corpus:
                centroid = info['centroid']

                # Find sentences with high similarity to centroid
                for sent in corpus[:100]:  # Sample for efficiency
                    sent_vec = self._sentence_vector(sent)
                    if sent_vec is not None:
                        sim = np.dot(centroid, sent_vec)
                        if sim > 0.5:  # Threshold
                            context_snippets.append((' '.join(sent[:20]), sim))

                context_snippets.sort(key=lambda x: x[1], reverse=True)
                context_snippets = context_snippets[:3]

            # Generate description
            desc = {
                'cluster_id': label,
                'size': info['size'],
                'concept_summary': f"Semantic region around: {', '.join(top_words[:3])}",
                'extended_neighbors': top_words,
                'coherence': info['mean_coherence'],
                'density': info['mean_density'],
                'context_examples': [s for s, _ in context_snippets],
                'likely_about': self._infer_theme(top_words)
            }

            descriptions.append(desc)

        self.cluster_descriptions = descriptions
        return descriptions

    def _sentence_vector(self, sent: List[str]) -> Optional[np.ndarray]:
        """Compute sentence vector as mean of word embeddings"""
        vecs = []
        for word in sent:
            vec = self.teacher.get_embedding(word)
            if vec is not None:
                vecs.append(vec)

        if not vecs:
            return None

        sent_vec = np.mean(vecs, axis=0)
        return sent_vec / (np.linalg.norm(sent_vec) + 1e-8)

    def _infer_theme(self, words: List[str]) -> str:
        """Infer theme from word list (simple heuristic)"""

        # Domain keywords
        themes = {
            'technology': ['computer', 'digital', 'software', 'system', 'data'],
            'biology': ['cell', 'gene', 'protein', 'life', 'organism'],
            'physics': ['energy', 'force', 'particle', 'wave', 'quantum'],
            'transportation': ['vehicle', 'wheel', 'road', 'travel', 'motion'],
            'animals': ['cat', 'dog', 'animal', 'creature', 'pet'],
            'literature': ['story', 'character', 'plot', 'narrative', 'theme']
        }

        # Check for theme matches
        for theme, keywords in themes.items():
            if any(any(kw in word for kw in keywords) for word in words):
                return f"Likely {theme}-related concepts"

        return "Mixed conceptual domain"

    def _compute_coherence(self, neighbors: List[Tuple[str, float]]) -> float:
        """Compute semantic coherence of neighbors"""
        if len(neighbors) < 2:
            return 0.0

        vecs = []
        for word, _ in neighbors[:10]:
            vec = self.teacher.get_embedding(word)
            if vec is not None:
                vecs.append(vec)

        if len(vecs) < 2:
            return 0.0

        # Average pairwise similarity
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(np.dot(vecs[i], vecs[j]))

        return float(np.mean(sims))

    def _compute_novelty(self, neighbors: List[Tuple[str, float]]) -> float:
        """Compute novelty as diversity of neighbors"""
        # Unique prefixes
        prefixes = set()
        for word, _ in neighbors[:10]:
            if len(word) >= 3:
                prefixes.add(word[:3])

        return len(prefixes) / max(len(neighbors[:10]), 1)

    def _compute_density(self, vec: np.ndarray, k: int = 50) -> float:
        """Compute local density around vector"""
        sims = self.teacher.embeddings @ vec
        top_sims = np.sort(sims)[-k:]
        return float(np.mean(top_sims))

    def stability_analysis(self, n_runs: int = 5) -> Dict:
        """Analyze stability across multiple mining runs"""

        print(f"\nRunning stability analysis ({n_runs} runs)...")

        all_centroids = []
        run_clusters = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")

            # Mine new batch
            self.batch_mine(n_probes=1000)

            # Cluster
            clusters = self.cluster_attractors(min_cluster_size=5)

            # Collect centroids
            for info in clusters.values():
                all_centroids.append(info['centroid'])
            run_clusters.append(len(clusters))

        # Analyze stability
        if len(all_centroids) > 10:
            # Cluster all centroids to find stable regions
            X = np.array(all_centroids)
            final_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=3,
                metric='euclidean'
            )
            labels = final_clusterer.fit_predict(X)

            n_stable = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            stability = {
                'n_stable_attractors': n_stable,
                'n_unstable': n_noise,
                'stability_score': 1 - (n_noise / len(labels)),
                'mean_clusters_per_run': np.mean(run_clusters),
                'std_clusters_per_run': np.std(run_clusters)
            }
        else:
            stability = {
                'n_stable_attractors': 0,
                'n_unstable': len(all_centroids),
                'stability_score': 0.0,
                'mean_clusters_per_run': np.mean(run_clusters) if run_clusters else 0,
                'std_clusters_per_run': np.std(run_clusters) if run_clusters else 0
            }

        return stability

    def save_results(self, filename: str = "mined_attractors.json"):
        """Save mining results to JSON"""

        results = {
            'n_attractors': len(self.mined_attractors),
            'n_clusters': len(self.clusters),
            'cluster_descriptions': self.cluster_descriptions,
            'top_attractors': []
        }

        # Add top attractors by coherence
        top_attractors = sorted(self.mined_attractors,
                               key=lambda x: x['coherence'],
                               reverse=True)[:10]

        for attr in top_attractors:
            results['top_attractors'].append({
                'neighbors': [w for w, _ in attr['neighbors'][:5]],
                'coherence': attr['coherence'],
                'novelty': attr['novelty'],
                'density': attr['density'],
                'strategy': attr['strategy']
            })

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filename}")


def run_mining_demo():
    """Demo of batch mining with auto-descriptions"""

    print("="*70)
    print("BATCH ATTRACTOR MINING DEMO")
    print("="*70)

    # Create and train a simple model
    from demo import create_toy_corpus

    corpus_texts = create_toy_corpus()

    trainer = MaskedAttractorTrainer(
        embed_dim=128,
        hidden_dim=256,
        use_transformer=False
    )

    corpus = trainer.prepare_corpus(corpus_texts)
    mask_set = {'bicycle', 'bike', 'helmet', 'wheel', 'pedal'}

    print("\nTraining model...")
    trainer.train(corpus, mask_set, epochs=30, batch_size=16)

    # Initialize miner
    miner = AttractorMiner(trainer)

    # Batch mine attractors
    print("\nBatch mining attractors...")
    attractors = miner.batch_mine(n_probes=2000)

    # Cluster
    clusters = miner.cluster_attractors(min_cluster_size=10)

    # Generate descriptions
    print("\nGenerating cluster descriptions...")
    descriptions = miner.generate_descriptions(corpus)

    print("\n" + "="*70)
    print("DISCOVERED ATTRACTOR CLUSTERS")
    print("="*70)

    for desc in descriptions:
        print(f"\nCluster {desc['cluster_id']} ({desc['size']} attractors)")
        print("-" * 50)
        print(f"Summary: {desc['concept_summary']}")
        print(f"Theme: {desc['likely_about']}")
        print(f"Coherence: {desc['coherence']:.3f}")
        print(f"Density: {desc['density']:.3f}")

        if desc['context_examples']:
            print(f"Context examples:")
            for ex in desc['context_examples'][:2]:
                print(f"  - {ex}")

    # Stability analysis
    stability = miner.stability_analysis(n_runs=3)

    print("\n" + "="*70)
    print("STABILITY ANALYSIS")
    print("="*70)
    print(f"Stable attractors: {stability['n_stable_attractors']}")
    print(f"Stability score: {stability['stability_score']:.2%}")
    print(f"Clusters per run: {stability['mean_clusters_per_run']:.1f} ± {stability['std_clusters_per_run']:.1f}")

    # Save results
    miner.save_results()

    print("\n" + "="*70)
    print("MINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    run_mining_demo()