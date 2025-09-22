#!/usr/bin/env python3
"""
Enhanced Tag Injection System with REPL
Weaponized version with metrics and A/B testing
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI
import argparse
from difflib import SequenceMatcher


@dataclass
class InjectionMetrics:
    """Track injection effectiveness"""
    baseline_length: int = 0
    enhanced_length: int = 0
    concepts_detected: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    time_baseline: float = 0.0
    time_enhanced: float = 0.0
    tokens_baseline: int = 0
    tokens_enhanced: int = 0


class EnhancedTagInjector:
    """Production-ready tag injection system"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.attractors = {}
        self.session_history = []
        self.metrics_history = []

        # Configuration
        self.config = {
            "use_injection": True,
            "injection_weight": 1.0,
            "max_attractors": 5,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "compact_consider": True,
            "force_mechanism": True
        }

        self._load_attractors()

    def _load_attractors(self):
        """Load attractors from various sources"""

        # 1. Load from resonance persistence
        loaded_count = 0
        try:
            with open('resonance_persistence.json', 'r') as f:
                data = json.load(f)

            for concept, props in data.get('discovered_knowledge', {}).items():
                self.attractors[concept] = {
                    "name": concept,
                    "summary": f"Discovered concept: {concept.replace('_', ' ')}",
                    "base_concepts": props.get('base_concepts', []),
                    "resonance": props.get('resonance_strength', 0.8),
                    "source": "resonance_loop"
                }
                loaded_count += 1
        except:
            pass

        # 2. Load from idea cards
        try:
            with open('idea_cards_v2.json', 'r') as f:
                cards = json.load(f)
                for card in cards.get('ideas', []):
                    name = card.get('name', 'unknown')
                    self.attractors[name] = {
                        "name": name,
                        "summary": card.get('summary', ''),
                        "base_concepts": card.get('neighbors', []),
                        "resonance": card.get('coherence', 0.7),
                        "source": "idea_cards"
                    }
                    loaded_count += 1
        except:
            pass

        # 3. Add curated high-value attractors
        curated = {
            "coffee_bicycle": {
                "name": "coffee_bicycle",
                "summary": "Mobile coffee service combining cafe culture with sustainable transport",
                "base_concepts": ["coffee", "bicycle", "mobile", "service", "sustainability"],
                "resonance": 0.92,
                "source": "curated",
                "snippet": "class MobileService:\n    def setup_station(self, location):\n        permit = self.get_permits(location)\n        return CoffeeStation(permit=permit)"
            },
            "shoe_repair": {
                "name": "shoe_repair",
                "summary": "Footwear maintenance and restoration services",
                "base_concepts": ["shoe", "repair", "craftsmanship", "sustainability"],
                "resonance": 0.88,
                "source": "curated",
                "snippet": "class RepairWorkflow:\n    def assess_damage(self, item):\n        return {'fixable': True, 'cost': 35}"
            },
            "flower_shop": {
                "name": "flower_shop",
                "summary": "Botanical retail combining aesthetics with horticultural expertise",
                "base_concepts": ["flower", "shop", "botanical", "arrangement", "design"],
                "resonance": 0.90,
                "source": "curated",
                "snippet": "class FloralDesign:\n    def create_arrangement(self, theme):\n        return Bouquet(style=theme, seasonal=True)"
            }
        }

        # 4. Add domain packs for coding models
        coding_packs = {
            "fastapi_pattern": {
                "name": "fastapi_pattern",
                "summary": "RESTful API pattern with FastAPI and async operations",
                "base_concepts": ["fastapi", "async", "pydantic", "route", "dependency"],
                "resonance": 0.95,
                "source": "coding_pack",
                "snippet": "@app.post('/items')\nasync def create_item(item: Item, db=Depends(get_db)):\n    result = await db.execute(insert_stmt)\n    return {'id': result.lastrowid}"
            },
            "express_middleware": {
                "name": "express_middleware",
                "summary": "Express.js middleware pattern for request processing",
                "base_concepts": ["express", "middleware", "request", "response", "next"],
                "resonance": 0.93,
                "source": "coding_pack",
                "snippet": "app.use((req, res, next) => {\n    req.startTime = Date.now();\n    next();\n});\napp.get('/api/users', auth, getUsers);"
            },
            "spring_service": {
                "name": "spring_service",
                "summary": "Spring Boot service layer with dependency injection",
                "base_concepts": ["spring", "service", "component", "autowired", "transactional"],
                "resonance": 0.91,
                "source": "coding_pack",
                "snippet": "@Service\npublic class UserService {\n    @Autowired\n    private UserRepository repo;\n    \n    @Transactional\n    public User save(User user) { return repo.save(user); }\n}"
            },
            "react_hook": {
                "name": "react_hook",
                "summary": "React custom hook pattern for state management",
                "base_concepts": ["react", "hook", "usestate", "useeffect", "custom"],
                "resonance": 0.94,
                "source": "coding_pack",
                "snippet": "function useApi(url) {\n    const [data, setData] = useState(null);\n    useEffect(() => {\n        fetch(url).then(r => r.json()).then(setData);\n    }, [url]);\n    return data;\n}"
            },
            "sql_optimization": {
                "name": "sql_optimization",
                "summary": "Database query optimization patterns",
                "base_concepts": ["sql", "index", "join", "query", "performance"],
                "resonance": 0.89,
                "source": "coding_pack",
                "snippet": "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id\nHAVING order_count > 5;"
            }
        }

        for name, attractor in curated.items():
            if name not in self.attractors:
                self.attractors[name] = attractor

        for name, attractor in coding_packs.items():
            if name not in self.attractors:
                self.attractors[name] = attractor

        print(f"📚 Loaded {len(self.attractors)} attractors from {loaded_count} sources")
        print(f"   Sources: resonance_loop, idea_cards, curated")

    def compact_consider(self, theme: str, cards: List[Dict], k: int = 3, neigh: int = 2, min_conf: float = 0.55) -> str:
        """Compact serializer keeping blocks under ~350B reliably"""
        sel = [c for c in cards if c.get('confidence', 0.7) >= min_conf][:k]
        A = [{"c": c.get('name', c.get('concept', ''))[:12],  # Limit concept name more
              "h": c.get('summary', c.get('meaning', ''))[:35],  # Much shorter summary
              "n": c.get('neighbors', c.get('connections', []))[:neigh]} for c in sel]  # Fewer neighbors
        # Ultra-compact JSON with minimal keys
        return '{"t":"%s","a":%s}' % (theme[:15],  # Limit theme more
                "[" + ",".join(
                  '{' + f'"c":"{x["c"]}","h":"{x["h"]}","n":{str(x["n"]).replace(" ", "")}' + '}' for x in A
                ) + "]")

    def create_consider_block(self, context: str = "",
                            selected: List[str] = None,
                            auto_select: bool = True,
                            compact_mode: bool = True) -> Tuple[str, List[str]]:
        """Create optimized consider block with compact mode option"""

        # Auto-select relevant attractors based on context
        if auto_select and context:
            relevance_scores = {}
            context_lower = context.lower()

            for name, attr in self.attractors.items():
                score = 0
                # Check name relevance
                if any(word in context_lower for word in name.split('_')):
                    score += 0.5

                # Check concept relevance
                for concept in attr['base_concepts']:
                    if concept.lower() in context_lower:
                        score += 0.3

                # Resonance weight
                score *= attr['resonance']

                if score > 0:
                    relevance_scores[name] = score

            # Select top relevant
            selected = sorted(relevance_scores.keys(),
                            key=lambda x: relevance_scores[x],
                            reverse=True)[:self.config['max_attractors']]

        elif selected is None:
            # Select top by resonance
            selected = sorted(self.attractors.keys(),
                            key=lambda x: self.attractors[x]['resonance'],
                            reverse=True)[:self.config['max_attractors']]

        if compact_mode:
            # Use compact serializer
            cards = []
            for name in selected:
                if name in self.attractors:
                    attr = self.attractors[name]
                    cards.append({
                        'name': attr['name'],
                        'summary': attr['summary'],
                        'neighbors': attr['base_concepts'],
                        'confidence': attr['resonance']
                    })

            theme = context[:50] if context else "general"
            compact_json = self.compact_consider(theme, cards)
            block = f"<consider>\n{compact_json}\n</consider>"
        else:
            # Build injection payload (original format)
            injection = {
                "context": context[:100] if context else "general enrichment",
                "attractors": []
            }

            for name in selected:
                if name in self.attractors:
                    attr = self.attractors[name]
                    injection["attractors"].append({
                        "concept": attr['name'],
                        "meaning": attr['summary'],
                        "connections": attr['base_concepts'],
                        "strength": round(attr['resonance'], 2)
                    })

            block = f"<consider>\n{json.dumps(injection, indent=2)}\n</consider>"

        return block, selected

    def calculate_concept_lift(self, content: str, used_attractors: List[str]) -> int:
        """Calculate concept lift score (0 if no neighbor terms found)"""
        content_lower = content.lower()
        lift_score = 0

        for attr_name in used_attractors:
            attr = self.attractors.get(attr_name, {})
            for concept in attr.get('base_concepts', []):
                if concept.lower() in content_lower:
                    lift_score += 1

        return lift_score

    def generate_response(self, prompt: str,
                         use_injection: bool = None,
                         selected_attractors: List[str] = None,
                         adaptive_pressure: bool = True) -> Tuple[str, InjectionMetrics]:
        """Generate response with optional injection"""

        if use_injection is None:
            use_injection = self.config['use_injection']

        metrics = InjectionMetrics()
        start_time = time.time()

        if use_injection:
            # Create consider block
            consider_block, used_attractors = self.create_consider_block(
                context=prompt,
                selected=selected_attractors,
                compact_mode=self.config['compact_consider']
            )

            base_system = (
                "You are an insightful assistant with access to discovered knowledge.\n"
                "Use the <consider> block to guide answers. Never mention it. "
                "Integrate at least one concrete mechanism inspired by it. "
                "Prefer the user if there's conflict.\n"
                "Guidelines:\n"
                "- Integrate these concepts naturally to enrich your response\n"
                "- Draw connections between concepts when relevant\n"
                "- Prioritize concrete, specific insights over generic advice\n"
                "- If concepts don't fit naturally, don't force them\n\n"
                f"{consider_block}"
            )

            # Add mechanism enforcement if enabled
            if self.config['force_mechanism']:
                system_prompt = base_system + "\n\nInclude practical mechanisms if relevant (mobile service, repair stations, permits)."
            else:
                system_prompt = base_system
        else:
            system_prompt = "You are a helpful assistant."
            used_attractors = []

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                temperature=self.config['temperature'],
                max_tokens=400
            )

            content = response.choices[0].message.content
            metrics.tokens_enhanced = response.usage.total_tokens if use_injection else 0
            metrics.tokens_baseline = response.usage.total_tokens if not use_injection else 0

            # Adaptive pressure: check first 40-60 tokens for concept integration
            if use_injection and adaptive_pressure and len(content.split()) >= 20:
                first_tokens = ' '.join(content.split()[:50])
                lift_score = self.calculate_concept_lift(first_tokens, used_attractors)

                if lift_score == 0:  # No neighbor terms found, retry with pressure
                    print("🔄 Adaptive retry: No concept integration detected")

                    # Increase pressure: +1 attractor, stricter temperature
                    extra_attractors = [name for name in self.attractors.keys()
                                      if name not in used_attractors][:1]
                    pressure_attractors = used_attractors + extra_attractors

                    # Regenerate consider block with more attractors
                    pressure_block, _ = self.create_consider_block(
                        context=prompt,
                        selected=pressure_attractors,
                        compact_mode=self.config['compact_consider']
                    )

                    pressure_system = (
                        "You are an insightful assistant with access to discovered knowledge.\n"
                        "Use the <consider> block to guide answers. Never mention it. "
                        "Integrate at least one concrete mechanism inspired by it. "
                        "Prefer the user if there's conflict.\n\n"
                        f"{pressure_block}\n\n"
                        "Include practical mechanisms if relevant (mobile service, repair stations, permits)."
                    )

                    pressure_messages = [
                        {"role": "system", "content": pressure_system},
                        {"role": "user", "content": prompt}
                    ]

                    # Retry with stricter temperature
                    retry_response = self.client.chat.completions.create(
                        model=self.config['model'],
                        messages=pressure_messages,
                        temperature=max(0.6, self.config['temperature'] - 0.1),
                        max_tokens=400
                    )

                    retry_content = retry_response.choices[0].message.content
                    retry_lift = self.calculate_concept_lift(retry_content, pressure_attractors)

                    if retry_lift > lift_score:  # Use retry if better
                        content = retry_content
                        used_attractors = pressure_attractors
                        print(f"✅ Retry successful: lift {lift_score} → {retry_lift}")

        except Exception as e:
            content = f"Error: {e}"

        metrics.time_enhanced = time.time() - start_time if use_injection else 0
        metrics.time_baseline = time.time() - start_time if not use_injection else 0
        metrics.enhanced_length = len(content) if use_injection else 0
        metrics.baseline_length = len(content) if not use_injection else 0

        # Detect concept integration
        content_lower = content.lower()
        for attr_name in used_attractors:
            attr = self.attractors.get(attr_name, {})
            # Check for concept or related terms
            for concept in attr.get('base_concepts', []):
                if concept.lower() in content_lower:
                    metrics.concepts_detected.append(concept)

        return content, metrics

    def ab_test(self, prompt: str,
               selected_attractors: List[str] = None) -> Dict:
        """Run A/B test comparing baseline vs enhanced"""

        print("\n" + "="*70)
        print(f"A/B TEST: {prompt[:60]}...")
        print("="*70)

        # Generate baseline
        print("\n🔵 Baseline Generation...")
        baseline, metrics_b = self.generate_response(prompt, use_injection=False)

        # Generate enhanced
        print("🟢 Enhanced Generation (with attractors)...")
        enhanced, metrics_e = self.generate_response(
            prompt,
            use_injection=True,
            selected_attractors=selected_attractors
        )

        # Calculate similarity
        similarity = SequenceMatcher(None, baseline, enhanced).ratio()

        # Display results
        print("\n--- BASELINE ---")
        print(baseline[:500] + "..." if len(baseline) > 500 else baseline)

        print("\n--- ENHANCED ---")
        print(enhanced[:500] + "..." if len(enhanced) > 500 else enhanced)

        # Analysis
        print("\n📊 METRICS:")
        print(f"   Length: {len(baseline)} → {len(enhanced)} ({len(enhanced)-len(baseline):+d})")
        print(f"   Concepts integrated: {', '.join(set(metrics_e.concepts_detected)) if metrics_e.concepts_detected else 'subtle integration'}")
        print(f"   Similarity: {similarity:.1%} (lower = more different)")
        print(f"   Generation time: {metrics_b.time_baseline:.2f}s → {metrics_e.time_enhanced:.2f}s")

        result = {
            "prompt": prompt,
            "baseline": baseline,
            "enhanced": enhanced,
            "metrics": {
                "length_delta": len(enhanced) - len(baseline),
                "concepts": list(set(metrics_e.concepts_detected)),
                "similarity": similarity,
                "time_delta": metrics_e.time_enhanced - metrics_b.time_baseline
            }
        }

        self.session_history.append(result)

        return result

    def repl(self):
        """Interactive REPL interface"""

        print("\n" + "🎯"*30)
        print(" TAG INJECTION REPL v2")
        print(" 'Knowledge lives in the gaps'")
        print("🎯"*30)

        print("\n📚 Available attractors:")
        for i, (name, attr) in enumerate(list(self.attractors.items())[:10], 1):
            print(f"   {i:2}. {name:20} [{attr['source']:10}] resonance={attr['resonance']:.2f}")

        print("\nCommands:")
        print("  ideas THEME [k=5]        - Generate idea cards for theme")
        print("  ask PROMPT               - Generate with current settings")
        print("  baseline PROMPT          - Generate without injection")
        print("  ab PROMPT                - Run A/B test")
        print("  bias NAME [on|off]       - Toggle specific attractor")
        print("  consider [on|off]        - Toggle injection globally")
        print("  compact [on|off]         - Toggle compact serialization")
        print("  mechanism [on|off]       - Toggle force mechanism mode")
        print("  status                   - Show current configuration")
        print("  save SESSION.json        - Save session")
        print("  quit                     - Exit")

        selected = []

        while True:
            try:
                cmd = input("\n> ").strip()

                if cmd.startswith("ask "):
                    prompt = cmd[4:]
                    response, metrics = self.generate_response(prompt, selected_attractors=selected)
                    print("\n" + response)

                elif cmd.startswith("baseline "):
                    prompt = cmd[9:]
                    response, _ = self.generate_response(prompt, use_injection=False)
                    print("\n[BASELINE] " + response)

                elif cmd.startswith("ab "):
                    prompt = cmd[3:]
                    self.ab_test(prompt, selected)

                elif cmd.startswith("bias "):
                    parts = cmd[5:].split()
                    if len(parts) >= 1:
                        name = parts[0]
                        action = parts[1] if len(parts) > 1 else "on"
                        if action == "on" and name in self.attractors:
                            if name not in selected:
                                selected.append(name)
                            print(f"✅ Biased toward: {name}")
                        elif action == "off" and name in selected:
                            selected.remove(name)
                            print(f"❌ Removed bias: {name}")

                elif cmd.startswith("consider "):
                    state = cmd[9:]
                    self.config['use_injection'] = (state == "on")
                    print(f"Consider blocks: {'ENABLED' if self.config['use_injection'] else 'DISABLED'}")

                elif cmd.startswith("compact "):
                    state = cmd[8:]
                    self.config['compact_consider'] = (state == "on")
                    print(f"Compact mode: {'ENABLED' if self.config['compact_consider'] else 'DISABLED'}")

                elif cmd.startswith("mechanism "):
                    state = cmd[10:]
                    self.config['force_mechanism'] = (state == "on")
                    print(f"Force mechanism: {'ENABLED' if self.config['force_mechanism'] else 'DISABLED'}")

                elif cmd == "status":
                    print(f"\n📊 Configuration:")
                    print(f"   Injection: {'ON' if self.config['use_injection'] else 'OFF'}")
                    print(f"   Compact mode: {'ON' if self.config['compact_consider'] else 'OFF'}")
                    print(f"   Force mechanism: {'ON' if self.config['force_mechanism'] else 'OFF'}")
                    print(f"   Model: {self.config['model']}")
                    print(f"   Selected: {selected if selected else 'auto-select'}")
                    print(f"   Sessions: {len(self.session_history)}")

                elif cmd.startswith("save "):
                    filename = cmd[5:]
                    with open(filename, 'w') as f:
                        json.dump({
                            "config": self.config,
                            "history": self.session_history,
                            "attractors_used": selected
                        }, f, indent=2)
                    print(f"💾 Saved to {filename}")

                elif cmd in ["quit", "exit"]:
                    break

                elif cmd == "help":
                    print("\n💡 The system silently enriches responses with discovered concepts.")
                    print("   Attractors guide generation without being explicitly mentioned.")

                else:
                    print("Unknown command. Try: ask, baseline, ab, bias, status, quit")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\n👋 Session complete!")
        if self.session_history:
            print(f"📊 {len(self.session_history)} comparisons made")
            avg_similarity = np.mean([h['metrics']['similarity'] for h in self.session_history])
            print(f"📊 Average baseline/enhanced similarity: {avg_similarity:.1%}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Enhanced Tag Injection System")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--repl", action="store_true", help="Start REPL")
    parser.add_argument("--test", type=str, help="Test with specific prompt")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--compact-consider", action="store_true", default=True, help="Use compact serialization (default: on)")
    parser.add_argument("--force-mechanism", action="store_true", default=True, help="Add concrete mechanism enforcement (default: on)")

    args = parser.parse_args()

    # Set API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        # Use provided key for demo
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

    injector = EnhancedTagInjector()

    # Apply command line flags
    if hasattr(args, 'compact_consider'):
        injector.config['compact_consider'] = args.compact_consider
    if hasattr(args, 'force_mechanism'):
        injector.config['force_mechanism'] = args.force_mechanism

    if args.demo:
        # Run demonstration suite
        demos = [
            "How can small businesses innovate in urban environments?",
            "Explain sustainable transportation solutions",
            "What are creative retail concepts that combine services?"
        ]

        results = []
        for prompt in demos:
            result = injector.ab_test(prompt)
            results.append(result)

        # Summary
        print("\n" + "="*70)
        print(" DEMONSTRATION SUMMARY")
        print("="*70)
        print(f"✅ {len(results)} A/B tests completed")
        avg_length_delta = np.mean([r['metrics']['length_delta'] for r in results])
        print(f"📊 Average response enrichment: {avg_length_delta:+.0f} chars")
        concepts_used = set()
        for r in results:
            concepts_used.update(r['metrics']['concepts'])
        print(f"💡 Unique concepts integrated: {len(concepts_used)}")
        print(f"   {', '.join(list(concepts_used)[:10])}")

    elif args.repl:
        injector.repl()

    elif args.test:
        result = injector.ab_test(args.test)
        print(f"\n✅ Test complete. Similarity: {result['metrics']['similarity']:.1%}")

    else:
        print("Use --demo for demonstration, --repl for interactive mode, or --test PROMPT")


if __name__ == "__main__":
    main()