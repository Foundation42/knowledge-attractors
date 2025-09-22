#!/usr/bin/env python3
"""
Ollama Test Harness with Optimized Knowledge Attractors
Features: Compact serialization, adaptive pressure, concept lift guardrail
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class OllamaConfig:
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7

class OptimizedOllamaInjector:
    """Optimized injector for Ollama models"""

    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.attractors = self._load_curated_attractors()

    def _load_curated_attractors(self) -> Dict:
        """Load curated attractors optimized for smaller models"""
        return {
            "coffee_bicycle": {
                "name": "coffee_bicycle",
                "summary": "Mobile coffee service combining cafe culture with sustainable transport",
                "base_concepts": ["coffee", "bicycle", "mobile", "service", "sustainability"],
                "resonance": 0.92,
                "snippet": "class MobileService:\n    def setup_station(self, location):\n        permit = self.get_permits(location)\n        return CoffeeStation(permit=permit)"
            },
            "micro_repair": {
                "name": "micro_repair",
                "summary": "Small-scale repair services for everyday items and electronics",
                "base_concepts": ["repair", "micro", "service", "electronics", "sustainability"],
                "resonance": 0.88,
                "snippet": "class RepairService:\n    def assess_item(self, item):\n        return {'fixable': True, 'cost': estimate_cost(item)}"
            },
            "api_gateway": {
                "name": "api_gateway",
                "summary": "Centralized API management with routing and authentication",
                "base_concepts": ["api", "gateway", "routing", "auth", "microservices"],
                "resonance": 0.94,
                "snippet": "@app.middleware('http')\nasync def auth_middleware(request, call_next):\n    token = request.headers.get('authorization')\n    return await call_next(request)"
            },
            "edge_caching": {
                "name": "edge_caching",
                "summary": "Distributed caching strategy for improved performance",
                "base_concepts": ["cache", "edge", "performance", "cdn", "distributed"],
                "resonance": 0.89,
                "snippet": "cache = Redis(host='edge-cache')\nresult = cache.get(key) or compute_and_cache(key)"
            }
        }

    def compact_consider(self, theme: str, cards: List[Dict], k: int = 3, neigh: int = 2) -> str:
        """Ultra-compact serializer for <350B blocks"""
        sel = cards[:k]
        A = [{"c": c["name"][:12],
              "h": c["summary"][:35],
              "n": c["base_concepts"][:neigh]} for c in sel]
        return '{"t":"%s","a":%s}' % (theme[:15],
                "[" + ",".join(
                  '{' + f'"c":"{x["c"]}","h":"{x["h"]}","n":{str(x["n"]).replace(" ", "")}' + '}' for x in A
                ) + "]")

    def select_attractors(self, context: str, k: int = 3) -> List[Dict]:
        """Select relevant attractors based on context"""
        scores = {}
        context_lower = context.lower()

        for name, attr in self.attractors.items():
            score = 0
            # Name relevance
            if any(word in context_lower for word in name.split('_')):
                score += 0.5
            # Concept relevance
            for concept in attr['base_concepts']:
                if concept.lower() in context_lower:
                    score += 0.3
            # Resonance weight
            score *= attr['resonance']

            if score > 0:
                scores[name] = score

        # Return top k attractors
        top_names = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        return [self.attractors[name] for name in top_names if name in self.attractors]

    def calculate_concept_lift(self, content: str, attractors: List[Dict]) -> int:
        """Calculate concept integration score"""
        content_lower = content.lower()
        lift = 0
        for attr in attractors:
            for concept in attr['base_concepts']:
                if concept.lower() in content_lower:
                    lift += 1
        return lift

    def generate_with_ollama(self, prompt: str, system: str = None) -> str:
        """Generate response using Ollama"""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('response', 'No response')
        except Exception as e:
            return f"Error: {e}"

    def enhanced_generate(self, prompt: str, adaptive_pressure: bool = True) -> Tuple[str, Dict]:
        """Generate with attractors and adaptive pressure"""

        # Select attractors
        attractors = self.select_attractors(prompt)

        if not attractors:
            # Fallback without injection
            return self.generate_with_ollama(prompt), {"lift": 0, "retries": 0}

        # Create compact consider block
        compact_json = self.compact_consider(prompt[:20], attractors)
        consider_block = f"<consider>\n{compact_json}\n</consider>"

        # System prompt with concrete mechanism contract
        system_prompt = (
            "Use the <consider> block to guide answers. Never mention it. "
            "Integrate at least one concrete mechanism inspired by it. "
            "Prefer the user if there's conflict.\n\n"
            f"{consider_block}\n\n"
            "Include practical mechanisms if relevant (mobile service, repair stations, permits)."
        )

        # First attempt
        content = self.generate_with_ollama(prompt, system_prompt)
        lift_score = self.calculate_concept_lift(content, attractors)

        # Adaptive pressure: retry if no concept integration
        retries = 0
        if adaptive_pressure and lift_score == 0 and len(content.split()) >= 20:
            print("🔄 Adaptive retry: No concept integration detected")

            # Add pressure: more attractors, stricter prompt
            extra_attractors = [attr for name, attr in self.attractors.items()
                              if attr not in attractors][:1]
            pressure_attractors = attractors + extra_attractors

            # Stronger pressure prompt
            pressure_json = self.compact_consider(prompt[:20], pressure_attractors)
            pressure_block = f"<consider>\n{pressure_json}\n</consider>"

            pressure_system = (
                "IMPORTANT: Use the <consider> block to guide answers. Never mention it. "
                "You MUST integrate at least one concrete mechanism inspired by it.\n\n"
                f"{pressure_block}\n\n"
                "Examples: mobile service stations, repair workshops, API endpoints, caching layers."
            )

            # Retry with pressure
            retry_content = self.generate_with_ollama(prompt, pressure_system)
            retry_lift = self.calculate_concept_lift(retry_content, pressure_attractors)

            if retry_lift > lift_score:
                content = retry_content
                lift_score = retry_lift
                attractors = pressure_attractors
                print(f"✅ Retry successful: lift {lift_score}")

            retries = 1

        return content, {
            "lift": lift_score,
            "retries": retries,
            "attractors": [a["name"] for a in attractors],
            "block_size": len(consider_block.encode('utf-8'))
        }

    def ab_test(self, prompt: str) -> Dict:
        """Run A/B test: baseline vs enhanced"""
        print(f"\n🧪 A/B Test: {prompt[:60]}...")
        print("=" * 70)

        # Baseline
        print("🔵 Baseline:")
        start_time = time.time()
        baseline = self.generate_with_ollama(prompt)
        baseline_time = time.time() - start_time
        print(f"   {baseline[:100]}...")

        # Enhanced
        print("\n🟢 Enhanced:")
        start_time = time.time()
        enhanced, metrics = self.enhanced_generate(prompt)
        enhanced_time = time.time() - start_time
        print(f"   {enhanced[:100]}...")

        # Analysis
        print(f"\n📊 Metrics:")
        print(f"   Length: {len(baseline)} → {len(enhanced)} ({len(enhanced)-len(baseline):+d})")
        print(f"   Concept lift: {metrics['lift']}")
        print(f"   Block size: {metrics['block_size']} bytes")
        print(f"   Attractors: {', '.join(metrics['attractors'])}")
        print(f"   Time: {baseline_time:.1f}s → {enhanced_time:.1f}s")
        print(f"   Retries: {metrics['retries']}")

        return {
            "prompt": prompt,
            "baseline": baseline,
            "enhanced": enhanced,
            "metrics": metrics,
            "length_delta": len(enhanced) - len(baseline),
            "time_delta": enhanced_time - baseline_time
        }

def main():
    """Demo the optimized injector"""

    print("🚀 Ollama Optimized Knowledge Attractor System")
    print("Features: Compact serialization, adaptive pressure, concept lift")
    print("=" * 70)

    injector = OptimizedOllamaInjector()

    # Test prompts targeting different domains
    test_prompts = [
        "How can small businesses serve coffee to busy urban commuters?",
        "What's the best way to build a scalable API for a mobile app?",
        "How can people extend the life of their electronics?",
        "Design a caching strategy for a global web application"
    ]

    results = []

    for prompt in test_prompts:
        result = injector.ab_test(prompt)
        results.append(result)
        time.sleep(1)  # Rate limiting

    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"✅ {len(results)} tests completed")

    avg_length_delta = sum(r['length_delta'] for r in results) / len(results)
    avg_lift = sum(r['metrics']['lift'] for r in results) / len(results)
    total_retries = sum(r['metrics']['retries'] for r in results)

    print(f"📈 Average response enrichment: {avg_length_delta:+.0f} chars")
    print(f"🎯 Average concept lift: {avg_lift:.1f}")
    print(f"🔄 Total adaptive retries: {total_retries}")

    # Show most effective test
    best_test = max(results, key=lambda x: x['metrics']['lift'])
    print(f"🏆 Best concept integration: '{best_test['prompt'][:40]}...' (lift: {best_test['metrics']['lift']})")

if __name__ == "__main__":
    main()