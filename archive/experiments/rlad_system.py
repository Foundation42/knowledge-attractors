#!/usr/bin/env python3
"""
RLAD: Reinforcement Learning for Active Discovery
A curiosity-driven training loop where the model asks for what it needs next
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import json
from masked_attractor import MaskedAttractorTrainer
from attractor_mining import AttractorMiner


@dataclass
class LearningQuery:
    """A query the student wants answered"""
    attractor_id: int
    vector: np.ndarray
    neighbors: List[str]
    uncertainty: float
    expected_value: float
    query_text: str


@dataclass
class LearningExperience:
    """Record of a learning interaction"""
    query: LearningQuery
    response: Dict[str, Any]
    reward: float
    state_before: np.ndarray
    state_after: np.ndarray
    metrics: Dict[str, float]


class CuriosityPolicy(nn.Module):
    """Policy network for selecting what to learn next"""

    def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and value estimate"""
        features = self.encoder(state)
        value = self.value_head(features)
        logits = self.policy_head(features)
        return logits, value

    def select_action(self, states: torch.Tensor,
                     epsilon: float = 0.1) -> Tuple[int, float]:
        """Select which attractor to explore"""

        with torch.no_grad():
            logits, values = self.forward(states)

            # Epsilon-greedy exploration
            if np.random.random() < epsilon:
                action = np.random.randint(len(logits))
            else:
                probs = F.softmax(logits.squeeze(), dim=0)
                action = torch.multinomial(probs, 1).item()

            return action, values[action].item()


class ActiveLearner:
    """Student that actively decides what to learn next"""

    def __init__(self, trainer: MaskedAttractorTrainer,
                 learning_rate: float = 1e-3):
        self.trainer = trainer
        self.miner = AttractorMiner(trainer)
        self.policy = CuriosityPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Learning memory
        self.experience_buffer = deque(maxlen=1000)
        self.knowledge_graph = {}  # Track what we've learned
        self.query_count = 0
        self.total_reward = 0.0

    def generate_queries(self, n_candidates: int = 20) -> List[LearningQuery]:
        """Generate candidate learning queries"""

        # Mine attractors
        attractors = self.miner.batch_mine(n_probes=n_candidates * 5)

        # Convert to queries
        queries = []
        for i, attr in enumerate(attractors[:n_candidates]):
            neighbors = [w for w, _ in attr['neighbors'][:5]]

            # Compute uncertainty (entropy of neighbor distribution)
            uncertainty = self._compute_uncertainty(attr['vector'])

            # Estimate expected value
            expected_value = self._estimate_value(attr)

            # Generate query text
            query_text = f"What connects {', '.join(neighbors[:3])}?"

            query = LearningQuery(
                attractor_id=i,
                vector=attr['vector'],
                neighbors=neighbors,
                uncertainty=uncertainty,
                expected_value=expected_value,
                query_text=query_text
            )
            queries.append(query)

        return queries

    def select_query(self, queries: List[LearningQuery],
                    epsilon: float = 0.1) -> LearningQuery:
        """Use policy to select which query to pursue"""

        # Convert queries to state tensors
        states = []
        for q in queries:
            state = self._query_to_state(q)
            states.append(state)

        states_tensor = torch.FloatTensor(np.array(states))

        # Select action using policy
        action, value = self.policy.select_action(states_tensor, epsilon)

        selected = queries[action]
        print(f"\n🎯 Selected query: {selected.query_text}")
        print(f"   Uncertainty: {selected.uncertainty:.3f}")
        print(f"   Expected value: {selected.expected_value:.3f}")

        return selected

    def ask_mentor(self, query: LearningQuery,
                  mentor_model: Optional[Any] = None) -> Dict[str, Any]:
        """Query mentor for knowledge"""

        # Simulate mentor response (in real system, call LLM or retrieval)
        response = {
            'definition': f"Concept combining {' and '.join(query.neighbors[:2])}",
            'examples': [
                f"Example involving {query.neighbors[0]}",
                f"Case study of {query.neighbors[1]}"
            ],
            'connections': query.neighbors[:5],
            'confidence': 0.8,
            'tokens_used': 50
        }

        self.query_count += 1
        return response

    def integrate_knowledge(self, query: LearningQuery,
                           response: Dict[str, Any]) -> Dict[str, float]:
        """Integrate new knowledge into student model"""

        # Update knowledge graph
        concept_id = f"concept_{self.query_count}"
        self.knowledge_graph[concept_id] = {
            'vector': query.vector,
            'neighbors': query.neighbors,
            'definition': response['definition'],
            'learned_at': self.query_count
        }

        # Compute integration metrics
        metrics = {
            'delta_h': self._compute_entropy_reduction(query.vector),
            'stability': self._compute_stability(query.vector),
            'novelty': self._compute_novelty(query.neighbors),
            'utility': np.random.random() * 0.5 + 0.5  # Simulated
        }

        return metrics

    def compute_reward(self, query: LearningQuery,
                      response: Dict[str, Any],
                      metrics: Dict[str, float]) -> float:
        """Compute reward for learning interaction"""

        # Composite reward function
        r = (
            0.3 * metrics['delta_h'] +           # Information gain
            0.2 * metrics['novelty'] +           # Off-manifold discovery
            0.2 * metrics['stability'] +          # Cross-run persistence
            0.2 * metrics['utility'] -           # Task improvement
            0.05 * response['tokens_used'] / 100 - # Cost penalty
            0.05 * self._compute_redundancy(query.neighbors)  # Redundancy penalty
        )

        return float(r)

    def update_policy(self, experience: LearningExperience):
        """Update curiosity policy using experience"""

        # Convert to tensors
        state_before = torch.FloatTensor(experience.state_before).unsqueeze(0)
        state_after = torch.FloatTensor(experience.state_after).unsqueeze(0)
        reward = torch.FloatTensor([experience.reward])

        # Compute TD error
        _, value_before = self.policy(state_before)
        _, value_after = self.policy(state_after)

        td_error = reward + 0.99 * value_after - value_before

        # Update policy (simplified, use PPO in production)
        loss = -td_error * value_before

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learning_loop(self, n_iterations: int = 10,
                     epsilon_start: float = 0.3) -> List[LearningExperience]:
        """Main active learning loop"""

        experiences = []
        epsilon = epsilon_start

        print("\n" + "="*70)
        print("ACTIVE LEARNING LOOP")
        print("="*70)

        for iteration in range(n_iterations):
            print(f"\n📖 Iteration {iteration + 1}/{n_iterations}")

            # 1. Generate candidate queries
            queries = self.generate_queries(n_candidates=10)

            # 2. Select query using policy
            selected = self.select_query(queries, epsilon=epsilon)

            # 3. Get current state
            state_before = self._get_global_state()

            # 4. Ask mentor
            response = self.ask_mentor(selected)
            print(f"   Mentor: {response['definition']}")

            # 5. Integrate knowledge
            metrics = self.integrate_knowledge(selected, response)

            # 6. Get new state
            state_after = self._get_global_state()

            # 7. Compute reward
            reward = self.compute_reward(selected, response, metrics)
            self.total_reward += reward
            print(f"   Reward: {reward:.3f} (total: {self.total_reward:.3f})")

            # 8. Create experience
            exp = LearningExperience(
                query=selected,
                response=response,
                reward=reward,
                state_before=state_before,
                state_after=state_after,
                metrics=metrics
            )
            experiences.append(exp)
            self.experience_buffer.append(exp)

            # 9. Update policy
            self.update_policy(exp)

            # 10. Decay exploration
            epsilon = max(0.1, epsilon * 0.95)

        return experiences

    def _query_to_state(self, query: LearningQuery) -> np.ndarray:
        """Convert query to state vector for policy"""
        # Combine query vector with metadata
        metadata = np.array([
            query.uncertainty,
            query.expected_value,
            len(query.neighbors) / 10.0,
            self.query_count / 100.0
        ])

        # Truncate/pad vector to fixed size
        vec = query.vector[:252] if len(query.vector) > 252 else \
              np.pad(query.vector, (0, 252 - len(query.vector)))

        state = np.concatenate([vec, metadata])
        return state

    def _get_global_state(self) -> np.ndarray:
        """Get current global learning state"""
        # Simplified: random state vector
        # In practice: encode knowledge graph, uncertainties, etc.
        return np.random.randn(256)

    def _compute_uncertainty(self, vector: np.ndarray) -> float:
        """Compute uncertainty for a region"""
        # Simplified: random
        # In practice: use cluster variance, NN distance distribution
        return np.random.random()

    def _estimate_value(self, attractor: Dict) -> float:
        """Estimate expected value of learning this attractor"""
        # Combine coherence and novelty
        return (attractor['coherence'] + attractor.get('novelty', 0.5)) / 2

    def _compute_entropy_reduction(self, vector: np.ndarray) -> float:
        """Compute reduction in uncertainty"""
        # Simplified
        return np.random.random() * 0.5 + 0.3

    def _compute_stability(self, vector: np.ndarray) -> float:
        """Compute cross-run stability"""
        # Use cached stability or compute
        return 0.8 + np.random.random() * 0.2

    def _compute_novelty(self, neighbors: List[str]) -> float:
        """Compute novelty vs existing knowledge"""
        # Check overlap with knowledge graph
        known_concepts = set()
        for concept in self.knowledge_graph.values():
            known_concepts.update(concept['neighbors'])

        overlap = len(set(neighbors) & known_concepts)
        return 1.0 - (overlap / max(len(neighbors), 1))

    def _compute_redundancy(self, neighbors: List[str]) -> float:
        """Compute redundancy with existing knowledge"""
        return 1.0 - self._compute_novelty(neighbors)

    def export_learning_diary(self, filename: str = "learning_diary.json"):
        """Export learning history for analysis"""

        diary = {
            'total_queries': self.query_count,
            'total_reward': float(self.total_reward),
            'knowledge_concepts': len(self.knowledge_graph),
            'experiences': []
        }

        for exp in list(self.experience_buffer)[-20:]:  # Last 20
            diary['experiences'].append({
                'query': exp.query.query_text,
                'response': exp.response['definition'],
                'reward': float(exp.reward),
                'metrics': exp.metrics
            })

        with open(filename, 'w') as f:
            json.dump(diary, f, indent=2)

        print(f"\n📚 Learning diary saved to {filename}")


def demo_active_learning():
    """Demo the active learning system"""

    print("="*70)
    print("RLAD: REINFORCEMENT LEARNING FOR ACTIVE DISCOVERY")
    print("="*70)

    # Create base model
    from demo import create_toy_corpus

    corpus_texts = create_toy_corpus()
    trainer = MaskedAttractorTrainer(embed_dim=128)

    corpus = trainer.prepare_corpus(corpus_texts)
    mask_set = {'bicycle', 'helmet', 'wheel'}

    print("\n📚 Training initial model...")
    trainer.train(corpus, mask_set, epochs=20, batch_size=16)

    # Initialize active learner
    learner = ActiveLearner(trainer)

    # Run active learning loop
    print("\n🚀 Starting active learning...")
    experiences = learner.learning_loop(n_iterations=5)

    # Summary
    print("\n" + "="*70)
    print("LEARNING SUMMARY")
    print("="*70)
    print(f"Total queries: {learner.query_count}")
    print(f"Total reward: {learner.total_reward:.3f}")
    print(f"Knowledge concepts: {len(learner.knowledge_graph)}")
    print(f"Average reward: {learner.total_reward / max(learner.query_count, 1):.3f}")

    # Show what was learned
    print("\n📝 Concepts learned:")
    for i, (cid, concept) in enumerate(learner.knowledge_graph.items(), 1):
        print(f"\n{i}. {concept['definition']}")
        print(f"   Key words: {', '.join(concept['neighbors'][:3])}")

    # Export diary
    learner.export_learning_diary()

    return learner


if __name__ == "__main__":
    learner = demo_active_learning()