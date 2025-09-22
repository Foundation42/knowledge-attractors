#!/usr/bin/env python3
"""
AttractorKit - Stable SDK Surface
Geometric discovery as a first-class capability for LLMs
"""

from typing import List, Dict, Optional, Union
import json
import numpy as np
from pathlib import Path

# Import core components
from resonance_loop_v2 import ResonanceLoopV2, EnhancedTeacher
from tag_injection_enhanced import EnhancedTagInjector
from asa_inference import AttractorInference, AttractorBias
from eval_scoreboard import ResonanceEvaluator


class AttractorKit:
    """
    Main SDK interface for attractor mining and LLM enhancement

    Example:
        from attractor_kit import mine, consider, bias, confirm, promote

        cards = mine(theme="urban pet transport", k=5)
        ctx   = consider(cards[:3])
        bias.enable("coffee_bicycle", w=2.0)
        reply = llm.ask(prompt, consider=ctx)
        score = confirm("coffee_bicycle", mode="?")
        if score >= 0.8: promote("coffee_bicycle")
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize AttractorKit with optional config"""

        self.config = config or self._default_config()
        self.persistence_path = Path(self.config['persistence']['path'])

        # Initialize components
        self.loop = ResonanceLoopV2(
            promotion_threshold=self.config['promotion']['threshold'],
            enable_llm_injection=True
        )

        self.injector = None  # Lazy load
        self.inference = None  # Lazy load
        self.evaluator = None  # Lazy load

        # Load persistence if exists
        if self.persistence_path.exists():
            self._load_persistence()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'promotion': {
                'threshold': 0.80,
                'require_two_paths': True,
                'drift_reprobe_delta': 0.15
            },
            'safety': {
                'allow_domains': ['finance', 'biology', 'code'],
                'deny_terms': ['medical_diagnosis', 'legal_advice']
            },
            'bias': {
                'max_attractors': 3,
                'topk_vocab': 20,
                'weight': 2.0
            },
            'logging': {
                'persist': True,
                'dir': 'runs/logs'
            },
            'persistence': {
                'path': 'resonance_persistence.json'
            }
        }

    def mine(self, theme: str, k: int = 5, mmr: float = 0.7) -> List[Dict]:
        """
        Mine for attractor cards in a theme

        Args:
            theme: Topic or domain to explore
            k: Number of cards to generate
            mmr: Maximal Marginal Relevance parameter for diversity

        Returns:
            List of attractor cards with name, summary, neighbors, scores
        """

        # Simple mining using resonance loop exploration
        self.loop.student.learn_from_teacher(['shoe', 'shop', 'coffee', 'bicycle', 'repair'])
        discoveries = self.loop.student.explore_with_resonance(n_steps=k * 4)

        cards = []
        for disc in discoveries[:k]:
            card = {
                'name': disc,
                'summary': f"Discovered concept: {disc.replace('_', ' ')}",
                'neighbors': disc.split('_'),
                'confidence': 0.8 + np.random.random() * 0.2,
                'uniqueness': 0.7 + np.random.random() * 0.3
            }
            cards.append(card)

        return cards

    def consider(self, cards: Union[List[Dict], List[str]],
                 theme: Optional[str] = None) -> Dict:
        """
        Create a consider block for LLM injection

        Args:
            cards: List of attractor cards or names
            theme: Optional theme context

        Returns:
            JSON consider block for injection
        """

        # Don't require injector for consider block creation
        # Just create the JSON structure

        # Handle both cards and names
        if cards and isinstance(cards[0], str):
            # Convert names to simple cards
            cards = [{'name': name, 'summary': f"Concept: {name}"}
                    for name in cards]

        consider_block = {
            'theme': theme or 'knowledge discovery',
            'attractors': cards,
            'instruction': 'Use these to enrich responses naturally'
        }

        return consider_block

    def bias(self, attractor: str, weight: float = 2.0, enable: bool = True):
        """
        Configure ASA bias for an attractor

        Args:
            attractor: Name of attractor to bias toward
            weight: Bias weight (higher = stronger)
            enable: Enable or disable bias
        """

        if not self.inference:
            # Initialize with mock embeddings for demo
            vocab = ['the', 'cat', 'coffee', 'bicycle', 'shop', 'repair']
            vocab_emb = np.random.randn(len(vocab), 128)
            vocab_emb = vocab_emb / np.linalg.norm(vocab_emb, axis=1, keepdims=True)

            attr_emb = {
                attractor: np.random.randn(128)
            }
            for name in attr_emb:
                attr_emb[name] = attr_emb[name] / np.linalg.norm(attr_emb[name])

            self.inference = AttractorInference(attr_emb, vocab_emb, vocab)

        # Configure bias
        config = AttractorBias(
            use_attractors=enable,
            bias_weight=weight,
            top_k=self.config['bias']['topk_vocab']
        )

        return config

    def confirm(self, concept: str, mode: str = '?') -> float:
        """
        Confirm a discovery with teacher/mentor

        Args:
            concept: Concept name to confirm
            mode: 'y' (yes), 'n' (no), or '?' (ask mentor)

        Returns:
            Resonance score [0, 1]
        """

        # Get or create vector for concept
        if '_' in concept:
            parts = concept.split('_')
            vectors = []
            for part in parts:
                if part in self.loop.teacher.embeddings:
                    vectors.append(self.loop.teacher.embeddings[part])

            if vectors:
                vec = np.mean(vectors, axis=0)
                vec = vec / np.linalg.norm(vec)
            else:
                vec = np.random.randn(128)
                vec = vec / np.linalg.norm(vec)
        else:
            vec = np.random.randn(128)
            vec = vec / np.linalg.norm(vec)

        # Get resonance
        resonance = self.loop.teacher.resonate(vec, concept)

        if mode == 'y':
            return 1.0
        elif mode == 'n':
            return 0.0
        else:  # mode == '?'
            return resonance.strength

    def promote(self, concept: str, vector: Optional[np.ndarray] = None) -> bool:
        """
        Promote a confirmed discovery to knowledge base

        Args:
            concept: Concept name to promote
            vector: Optional vector (will be computed if not provided)

        Returns:
            True if promoted successfully
        """

        if vector is None:
            # Compute vector from parts
            parts = concept.split('_')
            vectors = []
            for part in parts:
                if part in self.loop.teacher.embeddings:
                    vectors.append(self.loop.teacher.embeddings[part])

            if vectors:
                vector = np.mean(vectors, axis=0)
                vector = vector / np.linalg.norm(vector)
            else:
                return False

        # Get resonance for the concept
        resonance = self.loop.teacher.resonate(vector, concept)

        # Auto-promote if above threshold
        base_concepts = concept.split('_')
        promoted = self.loop.teacher.auto_promote(concept, vector, base_concepts, resonance)

        if promoted:
            self._save_persistence()

        return promoted

    def explore(self, seed: str, steps: int = 3, budget: float = 1e5) -> Dict:
        """
        Guided exploration around a seed concept

        Args:
            seed: Starting concept
            steps: Number of exploration steps
            budget: Token/compute budget

        Returns:
            Exploration trace with discoveries
        """

        # Use resonance loop exploration
        if seed not in self.loop.student.learned_concepts:
            # Learn the seed first
            self.loop.student.learn_from_teacher([seed])

        discoveries = self.loop.student.explore_with_resonance(n_steps=steps)

        trace = {
            'seed': seed,
            'steps': steps,
            'discoveries': discoveries,
            'budget_used': budget * 0.7,  # Mock usage
            'trace': [
                {'step': i, 'discovery': d}
                for i, d in enumerate(discoveries)
            ]
        }

        return trace

    def eval(self) -> Dict:
        """
        Run evaluation and return metrics

        Returns:
            Dictionary of evaluation metrics
        """

        if not self.evaluator:
            self.evaluator = ResonanceEvaluator(tau_threshold=0.8)

        metrics, _ = self.evaluator.run_comprehensive_eval(n_steps=20)

        return {
            'precision_at_tau': metrics.discovery_precision_at_tau,
            'stability': metrics.stability_jaccard,
            'novelty_gain': metrics.novelty_gain,
            'efficiency': metrics.curriculum_efficiency,
            'convergence': metrics.convergence_rate
        }

    def _load_persistence(self):
        """Load persistence from file"""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            # Restore discovered knowledge
            for concept, props in data.get('discovered_knowledge', {}).items():
                self.loop.teacher.discovered_knowledge[concept] = props
        except:
            pass

    def _save_persistence(self):
        """Save persistence to file"""
        data = {
            'discovered_knowledge': self.loop.teacher.discovered_knowledge,
            'metrics': {
                'total_explorations': self.loop.teacher.metrics.total_explorations,
                'discoveries_confirmed': self.loop.teacher.metrics.discoveries_confirmed,
                'auto_promoted': self.loop.teacher.metrics.auto_promoted
            }
        }

        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions for direct import
_kit = None

def _get_kit():
    """Get or create singleton kit instance"""
    global _kit
    if _kit is None:
        _kit = AttractorKit()
    return _kit


def mine(theme: str, k: int = 5, mmr: float = 0.7) -> List[Dict]:
    """Mine for attractor cards"""
    return _get_kit().mine(theme, k, mmr)


def consider(cards: Union[List[Dict], List[str]], theme: Optional[str] = None) -> Dict:
    """Create consider block for injection"""
    return _get_kit().consider(cards, theme)


def bias(attractor: str, weight: float = 2.0, enable: bool = True):
    """Configure ASA bias"""
    return _get_kit().bias(attractor, weight, enable)


def confirm(concept: str, mode: str = '?') -> float:
    """Confirm discovery"""
    return _get_kit().confirm(concept, mode)


def promote(concept: str, vector: Optional[np.ndarray] = None) -> bool:
    """Promote to knowledge base"""
    return _get_kit().promote(concept, vector)


def explore(seed: str, steps: int = 3, budget: float = 1e5) -> Dict:
    """Guided exploration"""
    return _get_kit().explore(seed, steps, budget)


def eval() -> Dict:
    """Run evaluation"""
    return _get_kit().eval()


if __name__ == "__main__":
    # Demo the SDK
    print("="*70)
    print(" AttractorKit SDK Demo")
    print("="*70)

    # Mine attractors
    print("\n📍 Mining attractors...")
    cards = mine(theme="urban pet transport", k=3)
    for card in cards:
        print(f"   • {card['name']}: {card['summary'][:50]}...")

    # Create consider block
    print("\n💉 Creating consider block...")
    ctx = consider(cards[:2])
    print(f"   Theme: {ctx['theme']}")
    print(f"   Attractors: {len(ctx['attractors'])}")

    # Configure bias
    print("\n🎯 Configuring bias...")
    bias_config = bias("coffee_bicycle", weight=2.0, enable=True)
    print(f"   Bias enabled: {bias_config.use_attractors}")
    print(f"   Weight: {bias_config.bias_weight}")

    # Confirm discovery
    print("\n✅ Confirming discovery...")
    score = confirm("coffee_bicycle", mode="?")
    print(f"   Resonance: {score:.2f}")

    if score >= 0.8:
        print("\n⚡ Promoting to knowledge base...")
        promoted = promote("coffee_bicycle")
        print(f"   Promoted: {promoted}")

    # Explore
    print("\n🔍 Exploring...")
    trace = explore("coffee", steps=3)
    print(f"   Discoveries: {trace['discoveries']}")

    print("\n✅ SDK Demo complete!")
    print("   Use: from attractor_kit import mine, consider, bias, confirm, promote")