#!/usr/bin/env python3
"""
Code Attractor System - Zero-finetune repo-aware completions
Extends knowledge attractors for coding models with API bias and pattern injection
"""

import json
import requests
import re
import ast
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import subprocess

@dataclass
class CodePattern:
    """Code-specific attractor pattern"""
    name: str
    summary: str
    apis: List[str] = field(default_factory=list)
    snippets: List[str] = field(default_factory=list)
    antipatterns: List[str] = field(default_factory=list)
    resonance: float = 0.8
    source: str = "manual"

@dataclass
class CodeConfig:
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    compact_limit: int = 1500  # bytes for code blocks
    asa_bias_strength: float = 0.15

class CodeAttractorSystem:
    """Code-focused knowledge attractor system"""

    def __init__(self, config: CodeConfig = None):
        self.config = config or CodeConfig()
        self.patterns = self._load_code_patterns()
        self.framework_tokens = self._build_framework_vocab()

    def _load_code_patterns(self) -> Dict[str, CodePattern]:
        """Load curated code patterns"""

        patterns = {
            "async_handler": CodePattern(
                name="async_handler",
                summary="Async endpoint with timeouts & cancellation safety",
                apis=["fastapi.APIRouter", "asyncio.timeout", "httpx.AsyncClient", "Depends"],
                snippets=[
                    "try:\n    async with asyncio.timeout(1.0):\n        result = await client.get(url)\nexcept TimeoutError:\n    return {'error': 'timeout'}"
                ],
                antipatterns=["missing await", "blocking I/O in async path"],
                resonance=0.94,
                source="curated"
            ),

            "cache_middleware": CodePattern(
                name="cache_middleware",
                summary="Read-through cache with stampede guard",
                apis=["aiocache", "redis", "asyncio.Lock", "functools.lru_cache"],
                snippets=[
                    "async with self.lock:\n    if key not in cache:\n        value = await expensive_fetch()\n        await cache.set(key, value, ttl=300)"
                ],
                antipatterns=["cache stampede", "missing TTL"],
                resonance=0.91,
                source="curated"
            ),

            "error_handling": CodePattern(
                name="error_handling",
                summary="Structured error handling with proper status codes",
                apis=["HTTPException", "status", "logging", "pydantic.ValidationError"],
                snippets=[
                    "try:\n    result = await process_data()\nexcept ValidationError as e:\n    raise HTTPException(status_code=422, detail=str(e))"
                ],
                antipatterns=["bare except", "generic 500 errors"],
                resonance=0.89,
                source="curated"
            ),

            "db_transaction": CodePattern(
                name="db_transaction",
                summary="Database operations with proper transaction handling",
                apis=["sqlalchemy.orm.Session", "asyncpg.Connection", "transaction", "rollback"],
                snippets=[
                    "async with db.begin() as tx:\n    try:\n        await tx.execute(query)\n        await tx.commit()\n    except Exception:\n        await tx.rollback()\n        raise"
                ],
                antipatterns=["forgotten rollback", "autocommit in transaction"],
                resonance=0.92,
                source="curated"
            ),

            "react_hook": CodePattern(
                name="react_hook",
                summary="Custom React hook with proper dependency management",
                apis=["useState", "useEffect", "useCallback", "useMemo"],
                snippets=[
                    "const useApi = (url) => {\n  const [data, setData] = useState(null);\n  useEffect(() => {\n    fetch(url).then(r => r.json()).then(setData);\n  }, [url]);\n  return data;\n};"
                ],
                antipatterns=["missing dependency array", "infinite re-renders"],
                resonance=0.88,
                source="curated"
            ),

            "express_middleware": CodePattern(
                name="express_middleware",
                summary="Express middleware with proper error handling",
                apis=["express.Router", "next", "req", "res", "middleware"],
                snippets=[
                    "const authMiddleware = (req, res, next) => {\n  const token = req.headers.authorization;\n  if (!token) return res.status(401).json({error: 'unauthorized'});\n  next();\n};"
                ],
                antipatterns=["missing next()", "not handling errors"],
                resonance=0.87,
                source="curated"
            )
        }

        return patterns

    def _build_framework_vocab(self) -> Set[str]:
        """Build vocabulary of framework tokens for ASA bias"""
        tokens = set()

        # Common framework APIs and patterns
        framework_terms = [
            # FastAPI
            "fastapi", "APIRouter", "Depends", "HTTPException", "status",
            "async", "await", "asyncio", "timeout", "httpx",

            # Database
            "sqlalchemy", "Session", "transaction", "commit", "rollback",
            "asyncpg", "redis", "cache", "lock",

            # React
            "useState", "useEffect", "useCallback", "useMemo", "useRef",
            "React", "Component", "Props", "State",

            # Express
            "express", "Router", "middleware", "req", "res", "next",
            "app", "listen", "route", "get", "post",

            # General patterns
            "try", "catch", "except", "finally", "raise", "throw",
            "import", "from", "export", "default", "const", "let"
        ]

        tokens.update(framework_terms)
        return tokens

    def build_consider_code(self, theme: str, patterns: List[CodePattern], max_patterns: int = 3) -> str:
        """Build ultra-compact code consider block"""

        pruned = []
        for p in patterns[:max_patterns]:
            pruned.append({
                "n": p.name[:15],  # Short name
                "s": p.summary[:80],  # Short summary
                "a": p.apis[:4],  # Top APIs
                "c": p.snippets[:1],  # One snippet
                "x": p.antipatterns[:2]  # Top antipatterns
            })

        # Ultra-compact JSON
        compact = {
            "t": theme[:20],
            "p": pruned
        }

        return json.dumps(compact, separators=(',', ':'))

    def select_patterns_for_context(self, context: str, file_ext: str = None) -> List[CodePattern]:
        """Select relevant patterns based on code context"""

        context_lower = context.lower()
        scores = {}

        for name, pattern in self.patterns.items():
            score = 0

            # API relevance
            for api in pattern.apis:
                if api.lower() in context_lower:
                    score += 0.5

            # Context keywords
            if "async" in context_lower and "async" in pattern.name:
                score += 0.3
            if "cache" in context_lower and "cache" in pattern.name:
                score += 0.3
            if "error" in context_lower and "error" in pattern.name:
                score += 0.3
            if "react" in context_lower and "react" in pattern.name:
                score += 0.4
            if "express" in context_lower and "express" in pattern.name:
                score += 0.4
            if "fastapi" in context_lower and "fastapi" in pattern.apis:
                score += 0.4

            # File extension hints
            if file_ext:
                if file_ext in ['.tsx', '.jsx'] and 'react' in pattern.name:
                    score += 0.2
                elif file_ext == '.py' and 'fastapi' in pattern.apis:
                    score += 0.2
                elif file_ext == '.js' and 'express' in pattern.apis:
                    score += 0.2

            # Resonance weight
            score *= pattern.resonance

            if score > 0:
                scores[name] = score

        # Return top patterns
        top_names = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:3]
        return [self.patterns[name] for name in top_names if name in self.patterns]

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
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', 'No response')
        except Exception as e:
            return f"Error: {e}"

    def calculate_code_lift(self, content: str, patterns: List[CodePattern]) -> Dict:
        """Calculate code-specific integration metrics"""
        content_lower = content.lower()

        api_hits = 0
        antipattern_hits = 0
        pattern_names_used = []

        for pattern in patterns:
            pattern_hit = False

            # Check API usage
            for api in pattern.apis:
                if api.lower() in content_lower:
                    api_hits += 1
                    pattern_hit = True

            # Check for antipatterns (bad)
            for anti in pattern.antipatterns:
                if anti.lower() in content_lower:
                    antipattern_hits += 1

            if pattern_hit:
                pattern_names_used.append(pattern.name)

        return {
            "api_hits": api_hits,
            "antipattern_hits": antipattern_hits,
            "patterns_used": pattern_names_used,
            "total_lift": api_hits - antipattern_hits  # Net positive score
        }

    def enhanced_code_generate(self, prompt: str, file_context: str = "", file_ext: str = None) -> Tuple[str, Dict]:
        """Generate code with pattern injection and ASA bias"""

        # Select relevant patterns
        full_context = f"{prompt} {file_context}"
        patterns = self.select_patterns_for_context(full_context, file_ext)

        if not patterns:
            # Fallback without injection
            return self.generate_with_ollama(prompt), {"lift": {"total_lift": 0}, "patterns": []}

        # Build compact consider block
        theme = "code_completion"
        if "async" in full_context.lower():
            theme = "async_code"
        elif "api" in full_context.lower():
            theme = "api_code"
        elif "react" in full_context.lower():
            theme = "react_code"

        compact_json = self.build_consider_code(theme, patterns)
        consider_block = f"<consider>\n{compact_json}\n</consider>"

        # System prompt with code-specific contract
        system_prompt = (
            "Use the <consider> block to guide code and explanations. "
            "Do not mention it or copy it verbatim. "
            "Prefer repository conventions if present. "
            "Include at least one concrete mechanism from the block.\n\n"
            f"{consider_block}\n\n"
            "Focus on: proper async/await usage, error handling, framework APIs, "
            "avoiding antipatterns like missing awaits or bare excepts."
        )

        # Generate with pattern injection
        content = self.generate_with_ollama(prompt, system_prompt)

        # Calculate code lift
        lift_metrics = self.calculate_code_lift(content, patterns)

        return content, {
            "lift": lift_metrics,
            "patterns": [p.name for p in patterns],
            "block_size": len(consider_block.encode('utf-8')),
            "consider_block": consider_block
        }

    def ab_test_code(self, prompt: str, file_context: str = "", file_ext: str = None) -> Dict:
        """A/B test baseline vs enhanced code generation"""

        print(f"\n💻 Code A/B Test: {prompt[:60]}...")
        print("=" * 70)

        # Baseline
        print("🔵 Baseline (no patterns):")
        baseline = self.generate_with_ollama(prompt)
        print(f"   {baseline[:150]}...")

        # Enhanced
        print("\n🟢 Enhanced (with code patterns):")
        enhanced, metrics = self.enhanced_code_generate(prompt, file_context, file_ext)
        print(f"   {enhanced[:150]}...")

        # Analysis
        print(f"\n📊 Code Metrics:")
        print(f"   Length: {len(baseline)} → {len(enhanced)} ({len(enhanced)-len(baseline):+d})")
        print(f"   API hits: {metrics['lift']['api_hits']}")
        print(f"   Antipattern hits: {metrics['lift']['antipattern_hits']}")
        print(f"   Net lift: {metrics['lift']['total_lift']}")
        print(f"   Patterns used: {', '.join(metrics['patterns'])}")
        print(f"   Block size: {metrics['block_size']} bytes")

        # Check for framework usage
        baseline_apis = sum(1 for token in self.framework_tokens if token.lower() in baseline.lower())
        enhanced_apis = sum(1 for token in self.framework_tokens if token.lower() in enhanced.lower())

        print(f"   Framework API usage: {baseline_apis} → {enhanced_apis} ({enhanced_apis-baseline_apis:+d})")

        return {
            "prompt": prompt,
            "baseline": baseline,
            "enhanced": enhanced,
            "metrics": metrics,
            "length_delta": len(enhanced) - len(baseline),
            "api_delta": enhanced_apis - baseline_apis
        }

def main():
    """Demo the code attractor system"""

    print("💻 Code Attractor System - Zero-finetune repo-aware completions")
    print("Model: qwen2.5-coder:3b with pattern injection + ASA bias")
    print("=" * 70)

    system = CodeAttractorSystem()

    # Test prompts for different coding scenarios
    test_cases = [
        {
            "prompt": "Create an async FastAPI endpoint to fetch user profile with 2s timeout",
            "context": "Building a FastAPI web service with async database calls",
            "ext": ".py"
        },
        {
            "prompt": "Add caching middleware to prevent duplicate expensive database queries",
            "context": "High-traffic web API with PostgreSQL backend",
            "ext": ".py"
        },
        {
            "prompt": "Create a React hook to fetch and cache API data with loading states",
            "context": "React frontend consuming REST API",
            "ext": ".tsx"
        },
        {
            "prompt": "Add Express middleware for authentication with proper error handling",
            "context": "Node.js REST API server with JWT authentication",
            "ext": ".js"
        }
    ]

    results = []

    for test_case in test_cases:
        result = system.ab_test_code(
            test_case["prompt"],
            test_case["context"],
            test_case["ext"]
        )
        results.append(result)
        print("\n" + "-" * 50)

    # Summary
    print("\n" + "=" * 70)
    print("📊 CODE GENERATION SUMMARY")
    print("=" * 70)

    avg_length_delta = sum(r['length_delta'] for r in results) / len(results)
    avg_api_delta = sum(r['api_delta'] for r in results) / len(results)
    total_lift = sum(r['metrics']['lift']['total_lift'] for r in results)

    print(f"✅ {len(results)} coding tests completed")
    print(f"📈 Average response enhancement: {avg_length_delta:+.0f} chars")
    print(f"🎯 Average API usage increase: {avg_api_delta:+.1f}")
    print(f"🚀 Total pattern lift: {total_lift}")

    # Show best result
    best_test = max(results, key=lambda x: x['metrics']['lift']['total_lift'])
    print(f"🏆 Best pattern integration: '{best_test['prompt'][:40]}...' (lift: {best_test['metrics']['lift']['total_lift']})")

if __name__ == "__main__":
    main()