#!/usr/bin/env python3
"""
Teacher-Student Resonance Loop
The student explores probabilistically based on attractors,
the teacher confirms when it resonates with actual knowledge
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class Resonance:
    """Teacher's resonance with student's exploration"""
    strength: float  # How strongly teacher recognizes this
    confirmation: str  # "yes, that's a thing" / "close but not quite" / "novel idea"
    enrichment: Dict  # Additional knowledge teacher provides
    examples: List[str]  # Concrete examples if it exists


class Teacher:
    """Teacher with rich knowledge who confirms student discoveries"""

    def __init__(self, knowledge_base: Dict[str, Dict]):
        """
        knowledge_base: Dictionary of concepts and their properties
        Example: {"shoe_shop": {"exists": True, "related": ["shoes", "retail", "shopping"]}}
        """
        self.knowledge_base = knowledge_base
        self.embeddings = {}  # Concept embeddings
        self.implicit_knowledge = {}  # Things teacher knows but hasn't taught

        # Build embeddings for known concepts
        self._build_embeddings()

    def _build_embeddings(self):
        """Create embeddings for known concepts"""
        n_concepts = len(self.knowledge_base)
        d = 128

        # Create structured embeddings (not random)
        for i, (concept, properties) in enumerate(self.knowledge_base.items()):
            # Base vector
            vec = np.random.randn(d) * 0.5

            # Add structure based on properties
            if 'category' in properties:
                # Concepts in same category are closer
                category_offset = hash(properties['category']) % 100
                vec[category_offset % d] += 0.5

            # Related concepts share dimensions
            if 'related' in properties:
                for related in properties['related']:
                    related_offset = hash(related) % d
                    vec[related_offset] += 0.2

            vec = vec / (np.linalg.norm(vec) + 1e-8)
            self.embeddings[concept] = vec

        # Also store implicit combinations (things that exist but weren't explicitly taught)
        self._generate_implicit_knowledge()

    def _generate_implicit_knowledge(self):
        """Generate implicit knowledge from combinations"""
        # Simple combinations that make sense
        combinations = [
            (["shoe", "shop"], "shoe_shop"),
            (["cat", "bicycle"], "cat_on_bicycle"),
            (["coffee", "shop"], "coffee_shop"),
            (["book", "store"], "bookstore"),
            (["flower", "shop"], "flower_shop"),
            (["repair", "shop"], "repair_shop"),
            (["online", "shop"], "online_shop"),
        ]

        for components, compound in combinations:
            if compound not in self.knowledge_base:
                self.implicit_knowledge[compound] = {
                    "components": components,
                    "exists": True,
                    "implicit": True,
                    "description": f"A {compound.replace('_', ' ')} - combination of {' and '.join(components)}"
                }

    def resonate(self, student_vector: np.ndarray,
                student_hypothesis: str = None) -> Resonance:
        """
        Check if student's exploration resonates with teacher's knowledge

        Args:
            student_vector: Where the student is exploring
            student_hypothesis: What the student thinks it found

        Returns:
            Resonance object with teacher's response
        """

        # Find nearest concepts in teacher's knowledge
        similarities = {}
        for concept, vec in self.embeddings.items():
            sim = np.dot(student_vector, vec)
            similarities[concept] = sim

        # Sort by similarity
        sorted_concepts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_concept, top_sim = sorted_concepts[0]

        # Check for implicit knowledge
        implicit_match = None
        if student_hypothesis:
            # Check if hypothesis matches implicit knowledge
            for implicit, properties in self.implicit_knowledge.items():
                if implicit in student_hypothesis or student_hypothesis in implicit:
                    implicit_match = implicit
                    break

        # Generate resonance response
        if top_sim > 0.8:
            # Strong resonance - student found something real
            confirmation = "Yes! That's definitely a thing."
            strength = top_sim

            if top_concept in self.knowledge_base:
                enrichment = self.knowledge_base[top_concept]
                examples = enrichment.get('examples', [f"A {top_concept.replace('_', ' ')}"])
            else:
                enrichment = {"discovered": True, "confirmed_by_resonance": True}
                examples = [f"That's like a {top_concept.replace('_', ' ')}"]

        elif top_sim > 0.6:
            # Moderate resonance
            confirmation = "You're onto something there."
            strength = top_sim
            enrichment = {"nearby_concept": top_concept, "similarity": top_sim}
            examples = [f"Similar to {top_concept.replace('_', ' ')}"]

        elif implicit_match:
            # Matches implicit knowledge
            confirmation = "Ah yes, that IS a thing! I hadn't explicitly mentioned it but you're right."
            strength = 0.9
            enrichment = self.implicit_knowledge[implicit_match]
            examples = [f"A {implicit_match.replace('_', ' ')} - good discovery!"]

        else:
            # Low resonance - might be novel
            confirmation = "That's a novel combination I hadn't considered."
            strength = top_sim
            enrichment = {"novel": True, "nearest": top_concept}
            examples = []

        return Resonance(
            strength=strength,
            confirmation=confirmation,
            enrichment=enrichment,
            examples=examples
        )

    def teach_explicit(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """Explicitly teach some concepts to student"""
        taught = {}
        for concept in concepts:
            if concept in self.embeddings:
                taught[concept] = self.embeddings[concept]
        return taught


class Student:
    """Student that explores based on what teacher has shown"""

    def __init__(self, teacher: Teacher):
        self.teacher = teacher
        self.learned_concepts = {}  # Explicitly taught
        self.discovered_concepts = {}  # Found through exploration
        self.exploration_history = []

    def learn_from_teacher(self, concepts: List[str]):
        """Learn explicit concepts from teacher"""
        taught = self.teacher.teach_explicit(concepts)
        self.learned_concepts.update(taught)
        print(f"📚 Learned {len(taught)} concepts: {', '.join(taught.keys())}")

    def explore_attractor(self, base_concepts: List[str]) -> Tuple[np.ndarray, str]:
        """
        Explore the space between taught concepts

        Returns:
            Attractor vector and hypothesis about what it might be
        """
        if not base_concepts:
            return None, None

        # Get vectors for base concepts
        vectors = []
        for concept in base_concepts:
            if concept in self.learned_concepts:
                vectors.append(self.learned_concepts[concept])

        if not vectors:
            return None, None

        # Explore the attractor (gap) between concepts
        if len(vectors) == 1:
            # Perturb single concept
            attractor = vectors[0] + np.random.randn(len(vectors[0])) * 0.1
        else:
            # Interpolate between concepts
            weights = np.random.dirichlet([1] * len(vectors))
            attractor = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)

        # Add exploration noise
        attractor = attractor + np.random.randn(len(attractor)) * 0.05
        attractor = attractor / (np.linalg.norm(attractor) + 1e-8)

        # Generate hypothesis (what student thinks it found)
        if len(base_concepts) == 2:
            hypothesis = f"{base_concepts[0]}_{base_concepts[1]}"
        else:
            hypothesis = f"combination_of_{'_'.join(base_concepts)}"

        return attractor, hypothesis

    def explore_with_resonance(self, n_steps: int = 20):
        """
        Explore with teacher feedback loop
        """
        print("\n🚀 Student beginning exploration based on taught concepts...")
        print(f"   Known concepts: {list(self.learned_concepts.keys())}")
        print()

        discoveries = []

        for step in range(n_steps):
            # Choose concepts to explore between
            n_base = np.random.randint(1, min(4, len(self.learned_concepts) + 1))
            base_concepts = list(np.random.choice(
                list(self.learned_concepts.keys()),
                size=min(n_base, len(self.learned_concepts)),
                replace=False
            ))

            # Explore attractor
            attractor, hypothesis = self.explore_attractor(base_concepts)

            if attractor is None:
                continue

            # Get teacher resonance
            resonance = self.teacher.resonate(attractor, hypothesis)

            # Record exploration
            self.exploration_history.append({
                'step': step,
                'base_concepts': base_concepts,
                'hypothesis': hypothesis,
                'resonance': resonance
            })

            # Process resonance
            print(f"📍 Step {step + 1}:")
            print(f"   Student explores: {' + '.join(base_concepts)}")
            print(f"   Student thinks: '{hypothesis}'")
            print(f"   Teacher: {resonance.confirmation}")

            if resonance.strength > 0.7:
                print(f"   💡 Resonance strength: {resonance.strength:.2f}")

                if resonance.examples:
                    print(f"   Examples: {resonance.examples[0]}")

                # Strong resonance - this is a real discovery
                if resonance.strength > 0.8 or 'implicit' in resonance.enrichment:
                    self.discovered_concepts[hypothesis] = {
                        'vector': attractor,
                        'base_concepts': base_concepts,
                        'confirmed': True,
                        'resonance': resonance.strength
                    }
                    discoveries.append(hypothesis)
                    print(f"   ⭐ DISCOVERY CONFIRMED!")

            elif resonance.enrichment.get('novel', False):
                print(f"   🔮 Novel combination - teacher hadn't considered this")

            print()

        return discoveries


class ResonanceLoop:
    """Full teacher-student resonance loop"""

    def __init__(self):
        # Create teacher's knowledge base
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
        }

        self.teacher = Teacher(knowledge_base)
        self.student = Student(self.teacher)

    def run_teaching_session(self):
        """Run a complete teaching and exploration session"""

        print("="*70)
        print("TEACHER-STUDENT RESONANCE LOOP")
        print("'You told me about shoes and shops, I thought shoe shops!'")
        print("="*70)

        # Phase 1: Teacher teaches some basic concepts
        print("\n📖 Phase 1: Teacher teaches basic concepts")
        print("-" * 50)

        initial_concepts = ["shoe", "shop", "cat", "bicycle", "coffee"]
        self.student.learn_from_teacher(initial_concepts)

        # Phase 2: Student explores and teacher confirms
        print("\n🔍 Phase 2: Student explores, teacher resonates")
        print("-" * 50)

        discoveries = self.student.explore_with_resonance(n_steps=15)

        # Phase 3: Summary
        print("\n" + "="*70)
        print("RESONANCE SESSION SUMMARY")
        print("="*70)
        print(f"Initial concepts taught: {len(initial_concepts)}")
        print(f"Discoveries confirmed: {len(discoveries)}")

        if discoveries:
            print(f"\n✅ Confirmed discoveries:")
            for d in discoveries:
                print(f"   - {d.replace('_', ' ')}")

        # Analyze resonance patterns
        high_resonance = [h for h in self.student.exploration_history
                         if h['resonance'].strength > 0.7]
        novel = [h for h in self.student.exploration_history
                if h['resonance'].enrichment.get('novel', False)]

        print(f"\n📊 Exploration statistics:")
        print(f"   High resonance explorations: {len(high_resonance)}")
        print(f"   Novel combinations found: {len(novel)}")

        return self.student.exploration_history


def demo_resonance():
    """Demo the teacher-student resonance loop"""

    loop = ResonanceLoop()
    history = loop.run_teaching_session()

    # Save exploration log
    log = []
    for h in history:
        log.append({
            'step': h['step'],
            'exploration': ' + '.join(h['base_concepts']),
            'hypothesis': h['hypothesis'],
            'teacher_response': h['resonance'].confirmation,
            'strength': h['resonance'].strength
        })

    with open('resonance_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print("\n📝 Exploration log saved to resonance_log.json")
    print("\nThe student learned to find concepts the teacher recognized but hadn't explicitly taught!")

    return loop


if __name__ == "__main__":
    loop = demo_resonance()