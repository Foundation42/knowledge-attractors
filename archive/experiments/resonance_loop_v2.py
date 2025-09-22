#!/usr/bin/env python3
"""
Enhanced Teacher-Student Resonance Loop v2
With auto-promotion, counterfactuals, curriculum learning, and teach-back
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
from collections import defaultdict
import time


@dataclass
class Resonance:
    """Teacher's resonance with student's exploration"""
    strength: float
    confirmation: str
    enrichment: Dict
    examples: List[str]
    counterfactuals: List[Dict] = field(default_factory=list)  # Near-misses
    teach_back: Optional[str] = None  # Micro-summary for reinforcement


@dataclass
class DiscoveryMetrics:
    """Track discovery success over time"""
    total_explorations: int = 0
    high_resonance_count: int = 0
    discoveries_confirmed: int = 0
    novel_combinations: int = 0
    auto_promoted: int = 0
    counterfactuals_generated: int = 0
    curriculum_updates: int = 0
    convergence_rate: float = 0.0
    discovery_efficiency: float = 0.0

    def update(self, resonance: Resonance, was_promoted: bool = False):
        """Update metrics based on resonance"""
        self.total_explorations += 1
        if resonance.strength > 0.7:
            self.high_resonance_count += 1
        if resonance.strength > 0.8:
            self.discoveries_confirmed += 1
        if resonance.enrichment.get('novel', False):
            self.novel_combinations += 1
        if was_promoted:
            self.auto_promoted += 1
        if resonance.counterfactuals:
            self.counterfactuals_generated += len(resonance.counterfactuals)

        # Calculate efficiency
        if self.total_explorations > 0:
            self.discovery_efficiency = self.discoveries_confirmed / self.total_explorations
            self.convergence_rate = self.high_resonance_count / self.total_explorations


class EnhancedTeacher:
    """Teacher with richer knowledge and auto-promotion capabilities"""

    def __init__(self, knowledge_base: Dict[str, Dict],
                 promotion_threshold: float = 0.8,
                 enable_teach_back: bool = True):
        self.knowledge_base = knowledge_base
        self.embeddings = {}
        self.implicit_knowledge = {}
        self.discovered_knowledge = {}  # Auto-promoted discoveries
        self.promotion_threshold = promotion_threshold
        self.enable_teach_back = enable_teach_back
        self.metrics = DiscoveryMetrics()

        self._build_embeddings()

    def _build_embeddings(self):
        """Create embeddings for known concepts"""
        d = 128

        for i, (concept, properties) in enumerate(self.knowledge_base.items()):
            vec = np.random.randn(d) * 0.5

            if 'category' in properties:
                category_offset = hash(properties['category']) % 100
                vec[category_offset % d] += 0.5

            if 'related' in properties:
                for related in properties['related']:
                    related_offset = hash(related) % d
                    vec[related_offset] += 0.2

            vec = vec / (np.linalg.norm(vec) + 1e-8)
            self.embeddings[concept] = vec

        self._generate_implicit_knowledge()

    def _generate_implicit_knowledge(self):
        """Generate implicit knowledge from combinations"""
        combinations = [
            (["shoe", "shop"], "shoe_shop"),
            (["cat", "bicycle"], "cat_on_bicycle"),
            (["coffee", "shop"], "coffee_shop"),
            (["book", "store"], "bookstore"),
            (["flower", "shop"], "flower_shop"),
            (["repair", "shop"], "repair_shop"),
            (["online", "shop"], "online_shop"),
            (["bicycle", "repair"], "bicycle_repair"),
            (["shoe", "repair"], "shoe_repair"),
            (["coffee", "break"], "coffee_break"),
        ]

        for components, compound in combinations:
            if compound not in self.knowledge_base:
                self.implicit_knowledge[compound] = {
                    "components": components,
                    "exists": True,
                    "implicit": True,
                    "description": f"A {compound.replace('_', ' ')}"
                }

                # Also create embedding for implicit knowledge
                if all(c in self.embeddings for c in components):
                    vec = np.mean([self.embeddings[c] for c in components], axis=0)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    self.embeddings[compound] = vec

    def _generate_counterfactuals(self, student_vector: np.ndarray,
                                 hypothesis: str) -> List[Dict]:
        """Generate near-miss counterfactuals for training"""
        counterfactuals = []

        # Perturbation-based counterfactuals
        for i in range(3):
            # Small random perturbation
            perturb = np.random.randn(len(student_vector)) * 0.1
            counter_vec = student_vector + perturb
            counter_vec = counter_vec / (np.linalg.norm(counter_vec) + 1e-8)

            # Check what this perturbation leads to
            similarities = {}
            for concept, vec in self.embeddings.items():
                sim = np.dot(counter_vec, vec)
                similarities[concept] = sim

            top_concept = max(similarities.items(), key=lambda x: x[1])[0]

            counterfactuals.append({
                "vector": counter_vec.tolist(),
                "hypothesis": f"not_{hypothesis}",
                "leads_to": top_concept,
                "distance": np.linalg.norm(perturb)
            })

        return counterfactuals

    def _create_teach_back(self, concept: str, resonance_strength: float) -> str:
        """Create micro-summary for reinforcement"""
        if resonance_strength > 0.9:
            return f"Excellent! {concept.replace('_', ' ')} is exactly right."
        elif resonance_strength > 0.8:
            return f"Good discovery - {concept.replace('_', ' ')} is indeed a thing."
        elif resonance_strength > 0.7:
            return f"You're close with {concept.replace('_', ' ')}."
        else:
            return f"Interesting exploration of {concept.replace('_', ' ')}."

    def resonate(self, student_vector: np.ndarray,
                student_hypothesis: str = None) -> Resonance:
        """Enhanced resonance with counterfactuals and teach-back"""

        # Find nearest concepts
        similarities = {}
        for concept, vec in self.embeddings.items():
            sim = np.dot(student_vector, vec)
            similarities[concept] = sim

        # Check discovered knowledge too
        for concept, properties in self.discovered_knowledge.items():
            if 'vector' in properties:
                sim = np.dot(student_vector, properties['vector'])
                similarities[f"discovered_{concept}"] = sim

        sorted_concepts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_concept, top_sim = sorted_concepts[0]

        # Check implicit knowledge
        implicit_match = None
        if student_hypothesis:
            for implicit, properties in self.implicit_knowledge.items():
                if implicit in student_hypothesis or student_hypothesis in implicit:
                    implicit_match = implicit
                    break

        # Generate resonance
        counterfactuals = []
        teach_back = None

        if top_sim > 0.8:
            confirmation = "Yes! That's definitely a thing."
            strength = top_sim

            if top_concept in self.knowledge_base:
                enrichment = self.knowledge_base[top_concept]
                examples = enrichment.get('examples', [f"A {top_concept.replace('_', ' ')}"])
            else:
                enrichment = {"discovered": True, "confirmed_by_resonance": True}
                examples = [f"That's like a {top_concept.replace('_', ' ')}"]

            # Generate counterfactuals for high-resonance discoveries
            counterfactuals = self._generate_counterfactuals(student_vector, student_hypothesis)

        elif top_sim > 0.6:
            confirmation = "You're onto something there."
            strength = top_sim
            enrichment = {"nearby_concept": top_concept, "similarity": top_sim}
            examples = [f"Similar to {top_concept.replace('_', ' ')}"]

        elif implicit_match:
            confirmation = "Ah yes, that IS a thing! Good discovery."
            strength = 0.9
            enrichment = self.implicit_knowledge[implicit_match]
            examples = [f"A {implicit_match.replace('_', ' ')} - well found!"]
            counterfactuals = self._generate_counterfactuals(student_vector, implicit_match)

        else:
            confirmation = "That's a novel combination I hadn't considered."
            strength = top_sim
            enrichment = {"novel": True, "nearest": top_concept}
            examples = []

        # Create teach-back if enabled
        if self.enable_teach_back and student_hypothesis:
            teach_back = self._create_teach_back(student_hypothesis, strength)

        # Update metrics
        resonance = Resonance(
            strength=strength,
            confirmation=confirmation,
            enrichment=enrichment,
            examples=examples,
            counterfactuals=counterfactuals,
            teach_back=teach_back
        )

        self.metrics.update(resonance)

        return resonance

    def auto_promote(self, concept: str, vector: np.ndarray,
                    base_concepts: List[str], resonance: Resonance) -> bool:
        """Auto-promote high-resonance discoveries to knowledge base"""
        if resonance.strength >= self.promotion_threshold:
            # Add to discovered knowledge
            self.discovered_knowledge[concept] = {
                "vector": vector,
                "base_concepts": base_concepts,
                "promoted_at": time.time(),
                "resonance_strength": resonance.strength,
                "category": "discovered",
                "related": base_concepts,
                "examples": resonance.examples
            }

            # Also add embedding for future resonance checks
            self.embeddings[f"discovered_{concept}"] = vector

            self.metrics.auto_promoted += 1
            return True
        return False

    def teach_explicit(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """Explicitly teach concepts including auto-promoted ones"""
        taught = {}

        # Original concepts
        for concept in concepts:
            if concept in self.embeddings:
                taught[concept] = self.embeddings[concept]

        # Include recently discovered concepts with lower weight
        for disc_concept, properties in self.discovered_knowledge.items():
            if 'vector' in properties:
                # Teach with 0.7 weight to indicate it's learned but newer
                taught[f"learned_{disc_concept}"] = properties['vector'] * 0.7

        return taught


class EnhancedStudent:
    """Student with curriculum learning and incremental updates"""

    def __init__(self, teacher: EnhancedTeacher,
                 exploration_temperature: float = 0.1,
                 curriculum_update_freq: int = 10):
        self.teacher = teacher
        self.learned_concepts = {}
        self.discovered_concepts = {}
        self.exploration_history = []
        self.counterfactual_buffer = []  # Store for training
        self.temperature = exploration_temperature
        self.curriculum_update_freq = curriculum_update_freq
        self.exploration_count = 0
        self.curriculum_version = 0

    def learn_from_teacher(self, concepts: List[str]):
        """Learn concepts including auto-promoted discoveries"""
        taught = self.teacher.teach_explicit(concepts)
        self.learned_concepts.update(taught)
        print(f"📚 Learned {len(taught)} concepts (including {len([k for k in taught if 'learned_' in k])} discoveries)")

    def update_curriculum(self):
        """Incremental curriculum update with discovered concepts"""
        # Get all auto-promoted discoveries
        new_concepts = []
        for concept, properties in self.teacher.discovered_knowledge.items():
            if concept not in self.discovered_concepts:
                new_concepts.append(concept)

        if new_concepts:
            # Re-learn including new discoveries
            print(f"\n📖 Curriculum update v{self.curriculum_version + 1}: Adding {len(new_concepts)} discoveries")
            base_concepts = [c for c in self.learned_concepts if not c.startswith('learned_')]
            self.learn_from_teacher(base_concepts[:5])  # Re-teach core + discoveries
            self.curriculum_version += 1
            self.teacher.metrics.curriculum_updates += 1

    def explore_attractor(self, base_concepts: List[str]) -> Tuple[np.ndarray, str]:
        """Explore with adaptive temperature"""
        if not base_concepts:
            return None, None

        vectors = []
        for concept in base_concepts:
            if concept in self.learned_concepts:
                vectors.append(self.learned_concepts[concept])

        if not vectors:
            return None, None

        # Adaptive exploration
        if len(vectors) == 1:
            attractor = vectors[0] + np.random.randn(len(vectors[0])) * self.temperature
        else:
            weights = np.random.dirichlet([1] * len(vectors))
            attractor = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)

        # Add exploration noise (reduced for discovered concepts)
        noise_scale = self.temperature * (0.5 if self.curriculum_version > 0 else 1.0)
        attractor = attractor + np.random.randn(len(attractor)) * noise_scale
        attractor = attractor / (np.linalg.norm(attractor) + 1e-8)

        # Generate hypothesis
        if len(base_concepts) == 2:
            hypothesis = f"{base_concepts[0]}_{base_concepts[1]}"
        else:
            hypothesis = f"combination_of_{'_'.join(base_concepts)}"

        return attractor, hypothesis

    def explore_with_resonance(self, n_steps: int = 20):
        """Enhanced exploration with auto-promotion and curriculum updates"""
        print("\n🚀 Enhanced exploration with auto-promotion...")
        print(f"   Initial concepts: {list(self.learned_concepts.keys())[:5]}...")
        print()

        discoveries = []

        for step in range(n_steps):
            # Curriculum update check
            if step > 0 and step % self.curriculum_update_freq == 0:
                self.update_curriculum()

            # Choose concepts to explore
            n_base = np.random.randint(1, min(4, len(self.learned_concepts) + 1))
            available = list(self.learned_concepts.keys())
            base_concepts = list(np.random.choice(
                available,
                size=min(n_base, len(available)),
                replace=False
            ))

            # Explore
            attractor, hypothesis = self.explore_attractor(base_concepts)

            if attractor is None:
                continue

            # Get resonance
            resonance = self.teacher.resonate(attractor, hypothesis)

            # Record
            self.exploration_history.append({
                'step': step,
                'base_concepts': base_concepts,
                'hypothesis': hypothesis,
                'resonance': resonance,
                'curriculum_version': self.curriculum_version
            })

            # Store counterfactuals for training
            if resonance.counterfactuals:
                self.counterfactual_buffer.extend(resonance.counterfactuals)

            # Display progress
            print(f"📍 Step {step + 1}:")
            print(f"   Exploring: {' + '.join(base_concepts)}")
            print(f"   Hypothesis: '{hypothesis}'")
            print(f"   Teacher: {resonance.confirmation}")

            if resonance.teach_back:
                print(f"   💭 {resonance.teach_back}")

            # Auto-promotion check
            if self.teacher.auto_promote(hypothesis, attractor, base_concepts, resonance):
                print(f"   ⚡ AUTO-PROMOTED to knowledge base!")
                discoveries.append(hypothesis)

                # Immediate curriculum mini-update
                if len(discoveries) % 3 == 0:
                    self.update_curriculum()

            elif resonance.strength > 0.7:
                print(f"   💡 Resonance: {resonance.strength:.2f}")

                if resonance.strength > 0.8:
                    self.discovered_concepts[hypothesis] = {
                        'vector': attractor,
                        'base_concepts': base_concepts,
                        'confirmed': True,
                        'resonance': resonance.strength
                    }
                    discoveries.append(hypothesis)
                    print(f"   ⭐ DISCOVERY CONFIRMED!")

            elif resonance.enrichment.get('novel', False):
                print(f"   🔮 Novel combination")

            print()

            self.exploration_count += 1

        return discoveries


class ResonanceLoopV2:
    """Enhanced resonance loop with full feature set"""

    def __init__(self, promotion_threshold: float = 0.8,
                 enable_llm_injection: bool = False):
        # Enhanced knowledge base
        knowledge_base = {
            "shoe": {"category": "footwear", "related": ["foot", "wear"], "examples": ["sneakers", "boots"]},
            "shop": {"category": "retail", "related": ["store", "buy"], "examples": ["store", "market"]},
            "cat": {"category": "animal", "related": ["pet", "feline"], "examples": ["kitten", "tabby"]},
            "bicycle": {"category": "vehicle", "related": ["ride", "wheel"], "examples": ["bike", "cycle"]},
            "coffee": {"category": "beverage", "related": ["drink", "caffeine"], "examples": ["espresso", "latte"]},
            "book": {"category": "media", "related": ["read", "pages"], "examples": ["novel", "textbook"]},
            "flower": {"category": "plant", "related": ["garden", "bloom"], "examples": ["rose", "tulip"]},
            "repair": {"category": "service", "related": ["fix", "maintain"], "examples": ["fix", "mend"]},
            "online": {"category": "digital", "related": ["internet", "web"], "examples": ["digital", "virtual"]},
            "store": {"category": "retail", "related": ["shop", "sell"], "examples": ["retailer", "outlet"]},
            "break": {"category": "pause", "related": ["rest", "stop"], "examples": ["pause", "rest"]},
        }

        self.teacher = EnhancedTeacher(knowledge_base, promotion_threshold)
        self.student = EnhancedStudent(self.teacher)
        self.enable_llm_injection = enable_llm_injection
        self.session_metrics = []

    def generate_llm_tags(self, discoveries: List[str]) -> str:
        """Generate tags for LLM injection"""
        if not self.enable_llm_injection or not discoveries:
            return ""

        tags = []
        for disc in discoveries[:5]:  # Top 5 discoveries
            concept = disc.replace('_', ' ')
            tags.append(f"<discovery>{concept}</discovery>")

        return f"\n<!-- Resonance Loop Discoveries -->\n" + "\n".join(tags)

    def run_advanced_session(self, initial_concepts: List[str] = None,
                            n_steps: int = 20):
        """Run complete enhanced session"""

        print("="*70)
        print("ENHANCED TEACHER-STUDENT RESONANCE LOOP v2")
        print("With auto-promotion, counterfactuals, and curriculum learning")
        print("="*70)

        # Phase 1: Initial teaching
        print("\n📖 Phase 1: Initial Teaching")
        print("-" * 50)

        if initial_concepts is None:
            initial_concepts = ["shoe", "shop", "cat", "bicycle", "coffee"]

        self.student.learn_from_teacher(initial_concepts)

        # Phase 2: Enhanced exploration
        print("\n🔍 Phase 2: Enhanced Exploration with Auto-Promotion")
        print("-" * 50)

        discoveries = self.student.explore_with_resonance(n_steps=n_steps)

        # Phase 3: Metrics and summary
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)

        metrics = self.teacher.metrics
        print(f"\n📊 Discovery Metrics:")
        print(f"   Total explorations: {metrics.total_explorations}")
        print(f"   High resonance (>0.7): {metrics.high_resonance_count}")
        print(f"   Confirmed discoveries: {metrics.discoveries_confirmed}")
        print(f"   Auto-promoted: {metrics.auto_promoted}")
        print(f"   Novel combinations: {metrics.novel_combinations}")
        print(f"   Counterfactuals generated: {metrics.counterfactuals_generated}")
        print(f"   Curriculum updates: {metrics.curriculum_updates}")
        print(f"   Discovery efficiency: {metrics.discovery_efficiency:.1%}")
        print(f"   Convergence rate: {metrics.convergence_rate:.1%}")

        if discoveries:
            print(f"\n✅ All discoveries ({len(discoveries)}):")
            for d in discoveries:
                resonance_data = next((h for h in self.student.exploration_history
                                      if h['hypothesis'] == d), None)
                if resonance_data:
                    strength = resonance_data['resonance'].strength
                    print(f"   - {d.replace('_', ' ')} (resonance: {strength:.2f})")

        # Auto-promoted concepts
        if self.teacher.discovered_knowledge:
            print(f"\n⚡ Auto-promoted to knowledge base:")
            for concept, props in self.teacher.discovered_knowledge.items():
                print(f"   - {concept.replace('_', ' ')} (from: {' + '.join(props['base_concepts'])})")

        # Counterfactual buffer
        if self.student.counterfactual_buffer:
            print(f"\n🔄 Counterfactuals ready for training: {len(self.student.counterfactual_buffer)}")

        # LLM injection tags
        if self.enable_llm_injection:
            tags = self.generate_llm_tags(discoveries)
            if tags:
                print(f"\n🏷️ LLM Injection Tags Generated:")
                print(tags)

        # Save detailed log
        self.save_session_log(discoveries)

        return {
            'discoveries': discoveries,
            'metrics': metrics,
            'auto_promoted': list(self.teacher.discovered_knowledge.keys()),
            'counterfactuals': len(self.student.counterfactual_buffer),
            'llm_tags': self.generate_llm_tags(discoveries) if self.enable_llm_injection else None
        }

    def save_session_log(self, discoveries: List[str]):
        """Save detailed session log"""
        log = {
            'timestamp': time.time(),
            'metrics': {
                'total_explorations': self.teacher.metrics.total_explorations,
                'discoveries_confirmed': self.teacher.metrics.discoveries_confirmed,
                'auto_promoted': self.teacher.metrics.auto_promoted,
                'efficiency': self.teacher.metrics.discovery_efficiency,
                'convergence': self.teacher.metrics.convergence_rate
            },
            'discoveries': discoveries,
            'auto_promoted_concepts': list(self.teacher.discovered_knowledge.keys()),
            'exploration_history': [
                {
                    'step': h['step'],
                    'exploration': ' + '.join(h['base_concepts']),
                    'hypothesis': h['hypothesis'],
                    'strength': h['resonance'].strength,
                    'confirmation': h['resonance'].confirmation,
                    'teach_back': h['resonance'].teach_back,
                    'curriculum_v': h.get('curriculum_version', 0)
                }
                for h in self.student.exploration_history
            ]
        }

        with open('resonance_log_v2.json', 'w') as f:
            json.dump(log, f, indent=2)

        print("\n📝 Detailed log saved to resonance_log_v2.json")


def demo_enhanced_resonance():
    """Demo the enhanced resonance loop"""

    # Create enhanced loop
    loop = ResonanceLoopV2(
        promotion_threshold=0.8,
        enable_llm_injection=True
    )

    # Run session
    results = loop.run_advanced_session(
        initial_concepts=["shoe", "shop", "cat", "bicycle", "coffee", "book", "repair"],
        n_steps=25
    )

    print("\n" + "="*70)
    print("RESONANCE LOOP V2 COMPLETE")
    print("="*70)
    print("The system now:")
    print("✓ Auto-promotes discoveries when resonance ≥ 0.8")
    print("✓ Generates counterfactuals for robust training")
    print("✓ Updates curriculum incrementally")
    print("✓ Provides teach-back micro-summaries")
    print("✓ Tracks comprehensive metrics")
    print("✓ Generates LLM injection tags")
    print("\nThe loop feels even more human: explore → confirm → promote → teach back → adapt!")

    return loop, results


if __name__ == "__main__":
    loop, results = demo_enhanced_resonance()