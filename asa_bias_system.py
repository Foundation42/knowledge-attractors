#!/usr/bin/env python3
"""
ASA Bias System for Code Attractors
Token-level steering for framework APIs and patterns (Ollama-compatible)
"""

import json
import requests
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from code_attractor_system import CodeAttractorSystem, CodePattern, CodeConfig

@dataclass
class ASAConfig:
    bias_strength: float = 0.15  # How much to boost framework terms
    seed_tokens: int = 8  # Number of API tokens to seed
    retry_threshold: float = 0.3  # Min API hit rate to avoid retry
    max_retries: int = 1

class ASABiasSystem(CodeAttractorSystem):
    """Enhanced code system with ASA bias for API steering"""

    def __init__(self, config: CodeConfig = None, asa_config: ASAConfig = None):
        super().__init__(config)
        self.asa_config = asa_config or ASAConfig()
        self.api_seeds = self._build_api_seeds()

    def _build_api_seeds(self) -> Dict[str, List[str]]:
        """Build API seed collections by framework"""
        return {
            "fastapi": [
                "@app.get", "@app.post", "Depends(", "HTTPException",
                "async def", "await", "APIRouter", "status_code"
            ],
            "express": [
                "app.get(", "app.post(", "req.body", "res.json(",
                "middleware", "next()", "router", "status("
            ],
            "react": [
                "useState(", "useEffect(", "useCallback(", "useMemo(",
                "const [", "setData", "useRef", "React.FC"
            ],
            "sqlalchemy": [
                "Session(", "session.query", "session.commit()", "session.rollback()",
                "async with", "transaction", "Base", "relationship"
            ],
            "asyncio": [
                "async def", "await", "asyncio.timeout", "asyncio.gather",
                "async with", "TimeoutError", "asyncio.Lock", "create_task"
            ]
        }

    def detect_framework_context(self, prompt: str, file_context: str = "") -> List[str]:
        """Detect which frameworks are relevant to the context"""
        full_context = f"{prompt} {file_context}".lower()
        frameworks = []

        framework_keywords = {
            "fastapi": ["fastapi", "api", "endpoint", "route", "pydantic"],
            "express": ["express", "node", "middleware", "req", "res"],
            "react": ["react", "component", "hook", "jsx", "tsx", "state"],
            "sqlalchemy": ["database", "db", "query", "orm", "sql"],
            "asyncio": ["async", "await", "timeout", "concurrent"]
        }

        for framework, keywords in framework_keywords.items():
            if any(keyword in full_context for keyword in keywords):
                frameworks.append(framework)

        return frameworks[:2]  # Limit to top 2 frameworks

    def create_api_seed_prefix(self, frameworks: List[str], patterns: List[CodePattern]) -> str:
        """Create API seed prefix to bias token selection"""
        seeds = []

        # Add framework-specific seeds
        for framework in frameworks:
            if framework in self.api_seeds:
                seeds.extend(self.api_seeds[framework][:3])

        # Add pattern-specific APIs
        for pattern in patterns:
            seeds.extend(pattern.apis[:2])

        # Remove duplicates and limit
        unique_seeds = list(dict.fromkeys(seeds))[:self.asa_config.seed_tokens]

        if not unique_seeds:
            return ""

        # Create subtle seed prefix
        seed_text = " ".join(unique_seeds[:4])
        return f"# Common APIs: {seed_text}\n"

    def calculate_api_hit_rate(self, content: str, frameworks: List[str], patterns: List[CodePattern]) -> float:
        """Calculate how well the content uses expected APIs"""
        content_lower = content.lower()
        expected_apis = set()

        # Collect expected APIs
        for framework in frameworks:
            if framework in self.api_seeds:
                expected_apis.update(api.lower() for api in self.api_seeds[framework])

        for pattern in patterns:
            expected_apis.update(api.lower() for api in pattern.apis)

        if not expected_apis:
            return 1.0

        # Count hits
        hits = sum(1 for api in expected_apis if api in content_lower)
        hit_rate = hits / min(len(expected_apis), 8)  # Normalize to reasonable expectation

        return min(hit_rate, 1.0)

    def enhanced_generate_with_asa(self, prompt: str, file_context: str = "", file_ext: str = None) -> Tuple[str, Dict]:
        """Generate code with ASA bias and pattern injection"""

        # Detect frameworks and select patterns
        frameworks = self.detect_framework_context(prompt, file_context)
        full_context = f"{prompt} {file_context}"
        patterns = self.select_patterns_for_context(full_context, file_ext)

        # Create API seed prefix
        api_seed = self.create_api_seed_prefix(frameworks, patterns)

        # Build compact consider block
        theme = "code_completion"
        if frameworks:
            theme = f"{frameworks[0]}_code"

        compact_json = self.build_consider_code(theme, patterns) if patterns else ""
        consider_block = f"<consider>\n{compact_json}\n</consider>" if compact_json else ""

        # Enhanced system prompt with ASA bias
        system_prompt = (
            "You are an expert programmer. Use the <consider> block to guide code and explanations. "
            "Do not mention it or copy it verbatim. "
            "Prefer repository conventions if present. "
            "Include at least one concrete mechanism from the block.\n\n"
        )

        if consider_block:
            system_prompt += f"{consider_block}\n\n"

        system_prompt += (
            "Focus on: proper async/await usage, framework APIs, error handling, "
            "avoiding antipatterns. Use appropriate framework patterns and APIs."
        )

        # Create biased prompt with API seeds
        biased_prompt = api_seed + prompt if api_seed else prompt

        # First attempt
        content = self.generate_with_ollama(biased_prompt, system_prompt)

        # Calculate metrics
        lift_metrics = self.calculate_code_lift(content, patterns)
        api_hit_rate = self.calculate_api_hit_rate(content, frameworks, patterns)

        retries = 0

        # ASA retry if API hit rate is too low
        if (api_hit_rate < self.asa_config.retry_threshold and
            retries < self.asa_config.max_retries and
            len(content.split()) >= 20):

            print(f"🔄 ASA retry: API hit rate {api_hit_rate:.2f} below threshold {self.asa_config.retry_threshold}")

            # Stronger bias prompt
            stronger_seed = self.create_api_seed_prefix(frameworks, patterns)
            if stronger_seed:
                stronger_seed = stronger_seed.replace("# Common APIs:", "# Required APIs to use:")

            stronger_system = (
                "You are an expert programmer. IMPORTANT: Use the <consider> block patterns. "
                "You MUST use the appropriate framework APIs and patterns shown.\n\n"
            )

            if consider_block:
                stronger_system += f"{consider_block}\n\n"

            stronger_system += (
                f"Focus on these frameworks: {', '.join(frameworks)}\n"
                "Use their standard APIs, patterns, and best practices."
            )

            stronger_prompt = stronger_seed + prompt

            # Retry with stronger bias
            retry_content = self.generate_with_ollama(stronger_prompt, stronger_system)
            retry_hit_rate = self.calculate_api_hit_rate(retry_content, frameworks, patterns)
            retry_lift = self.calculate_code_lift(retry_content, patterns)

            if retry_hit_rate > api_hit_rate or retry_lift["total_lift"] > lift_metrics["total_lift"]:
                content = retry_content
                api_hit_rate = retry_hit_rate
                lift_metrics = retry_lift
                print(f"✅ ASA retry successful: hit rate {api_hit_rate:.2f}")

            retries = 1

        return content, {
            "lift": lift_metrics,
            "api_hit_rate": api_hit_rate,
            "frameworks": frameworks,
            "patterns": [p.name for p in patterns],
            "retries": retries,
            "block_size": len(consider_block.encode('utf-8')) if consider_block else 0,
            "api_seed": api_seed.strip() if api_seed else None
        }

    def ab_test_asa(self, prompt: str, file_context: str = "", file_ext: str = None) -> Dict:
        """A/B test: baseline vs ASA-biased generation"""

        print(f"\n🎯 ASA Bias A/B Test: {prompt[:60]}...")
        print("=" * 70)

        # Baseline (no ASA bias)
        print("🔵 Baseline (no ASA bias):")
        baseline, baseline_metrics = self.enhanced_code_generate(prompt, file_context, file_ext)
        baseline_hit_rate = self.calculate_api_hit_rate(
            baseline,
            self.detect_framework_context(prompt, file_context),
            self.select_patterns_for_context(f"{prompt} {file_context}", file_ext)
        )
        print(f"   {baseline[:150]}...")

        # ASA-biased
        print("\n🟢 ASA-Biased (framework steering):")
        asa_enhanced, asa_metrics = self.enhanced_generate_with_asa(prompt, file_context, file_ext)
        print(f"   {asa_enhanced[:150]}...")

        # Analysis
        print(f"\n📊 ASA Bias Metrics:")
        print(f"   Length: {len(baseline)} → {len(asa_enhanced)} ({len(asa_enhanced)-len(baseline):+d})")
        print(f"   API hit rate: {baseline_hit_rate:.2f} → {asa_metrics['api_hit_rate']:.2f}")
        print(f"   Net lift: {baseline_metrics['lift']['total_lift']} → {asa_metrics['lift']['total_lift']}")
        print(f"   Frameworks detected: {', '.join(asa_metrics['frameworks'])}")
        print(f"   ASA retries: {asa_metrics['retries']}")

        if asa_metrics['api_seed']:
            print(f"   API seeds used: {asa_metrics['api_seed'][:100]}...")

        return {
            "prompt": prompt,
            "baseline": baseline,
            "asa_enhanced": asa_enhanced,
            "baseline_metrics": baseline_metrics,
            "asa_metrics": asa_metrics,
            "api_improvement": asa_metrics['api_hit_rate'] - baseline_hit_rate,
            "lift_improvement": asa_metrics['lift']['total_lift'] - baseline_metrics['lift']['total_lift']
        }

def main():
    """Demo ASA bias system"""

    print("🎯 ASA Bias System - Token-level Framework Steering")
    print("Model: qwen2.5-coder:3b with API bias + pattern injection")
    print("=" * 70)

    asa_system = ASABiasSystem()

    # Test cases that should benefit from ASA bias
    test_cases = [
        {
            "prompt": "Create a user registration endpoint with validation",
            "context": "FastAPI web service with Pydantic models",
            "ext": ".py"
        },
        {
            "prompt": "Add authentication middleware with JWT token validation",
            "context": "Express.js REST API server",
            "ext": ".js"
        },
        {
            "prompt": "Create a data fetching hook with loading and error states",
            "context": "React TypeScript application",
            "ext": ".tsx"
        },
        {
            "prompt": "Implement database transaction with proper rollback handling",
            "context": "Python async application with SQLAlchemy",
            "ext": ".py"
        }
    ]

    results = []

    for test_case in test_cases:
        result = asa_system.ab_test_asa(
            test_case["prompt"],
            test_case["context"],
            test_case["ext"]
        )
        results.append(result)
        print("\n" + "-" * 50)

    # Summary
    print("\n" + "=" * 70)
    print("📊 ASA BIAS EFFECTIVENESS SUMMARY")
    print("=" * 70)

    avg_api_improvement = sum(r['api_improvement'] for r in results) / len(results)
    avg_lift_improvement = sum(r['lift_improvement'] for r in results) / len(results)
    total_retries = sum(r['asa_metrics']['retries'] for r in results)

    print(f"✅ {len(results)} ASA bias tests completed")
    print(f"🎯 Average API hit rate improvement: {avg_api_improvement:+.2f}")
    print(f"🚀 Average lift improvement: {avg_lift_improvement:+.1f}")
    print(f"🔄 Total ASA retries triggered: {total_retries}")

    # Show most effective result
    best_result = max(results, key=lambda x: x['api_improvement'])
    print(f"🏆 Best API steering: '{best_result['prompt'][:40]}...' (+{best_result['api_improvement']:.2f} hit rate)")

    # Show bias effectiveness
    positive_improvements = sum(1 for r in results if r['api_improvement'] > 0)
    print(f"📈 ASA bias helped in {positive_improvements}/{len(results)} cases")

if __name__ == "__main__":
    main()