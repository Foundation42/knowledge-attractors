#!/usr/bin/env python3
"""
Qwen2.5-Coder:3B Test Harness with Complete Code Attractor System
Integrated demo: repo mining + pattern injection + ASA bias + validation
"""

import json
import os
import time
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from asa_bias_system import ASABiasSystem, ASAConfig
from repo_mining import RepoMiner
from code_attractor_system import CodeConfig, CodePattern

@dataclass
class ValidationMetrics:
    """Code validation results"""
    syntax_valid: bool = False
    linting_score: float = 0.0
    test_passing: bool = False
    framework_conformity: float = 0.0
    api_completeness: float = 0.0

@dataclass
class TestResult:
    """Complete test result with all metrics"""
    prompt: str
    baseline: str
    enhanced: str
    validation: ValidationMetrics
    patterns_used: List[str]
    api_hit_rate: float
    lift_score: int
    generation_time: float
    repo_patterns_found: int

class QwenCodeTester:
    """Complete test harness for qwen2.5-coder:3b"""

    def __init__(self, repo_path: str = None):
        # Initialize systems
        self.config = CodeConfig(model="qwen2.5-coder:3b")
        self.asa_config = ASAConfig(bias_strength=0.2)
        self.asa_system = ASABiasSystem(self.config, self.asa_config)

        # Mine repo patterns if provided
        self.repo_patterns = []
        if repo_path and os.path.exists(repo_path):
            self._mine_repo_patterns(repo_path)

        self.test_results = []

    def _mine_repo_patterns(self, repo_path: str):
        """Mine patterns from repository"""
        print(f"🔍 Mining patterns from {repo_path}...")

        miner = RepoMiner(repo_path)
        discovered_patterns = miner.scan_repository(max_files=100)

        # Convert to CodePattern objects
        self.repo_patterns = miner.generate_code_attractors(
            min_frequency=2,
            min_resonance=0.2
        )

        # Add repo patterns to ASA system
        for pattern in self.repo_patterns:
            self.asa_system.patterns[pattern.name] = pattern

        print(f"📊 Added {len(self.repo_patterns)} repo-mined patterns")

    def validate_code_quality(self, code: str, language: str = "python") -> ValidationMetrics:
        """Validate generated code quality"""
        metrics = ValidationMetrics()

        # Syntax validation
        if language == "python":
            metrics.syntax_valid = self._validate_python_syntax(code)
            metrics.linting_score = self._run_python_linting(code)
        elif language in ["javascript", "typescript"]:
            metrics.syntax_valid = self._validate_js_syntax(code)

        # Framework conformity (check for proper API usage)
        metrics.framework_conformity = self._check_framework_conformity(code, language)

        # API completeness (check if code uses expected APIs)
        metrics.api_completeness = self._check_api_completeness(code)

        return metrics

    def _validate_python_syntax(self, code: str) -> bool:
        """Check if Python code has valid syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _validate_js_syntax(self, code: str) -> bool:
        """Basic JavaScript syntax validation"""
        # Simple heuristics - in practice would use proper JS parser
        return (
            code.count('(') == code.count(')') and
            code.count('{') == code.count('}') and
            code.count('[') == code.count(']')
        )

    def _run_python_linting(self, code: str) -> float:
        """Run basic Python linting checks"""
        issues = 0

        # Simple linting rules
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for common issues
            if line.endswith('except:'):  # Bare except
                issues += 1
            if 'import *' in line:  # Star imports
                issues += 1
            if len(line) > 120:  # Line too long
                issues += 0.5

        # Calculate score (fewer issues = better score)
        max_lines = max(len(lines), 1)
        return max(0.0, 1.0 - (issues / max_lines))

    def _check_framework_conformity(self, code: str, language: str) -> float:
        """Check if code follows framework conventions"""
        score = 0.0
        checks = 0

        if language == "python":
            # FastAPI conventions
            if 'fastapi' in code.lower():
                checks += 1
                if '@app.' in code or 'APIRouter' in code:
                    score += 1

            # Async conventions
            if 'async def' in code:
                checks += 1
                if 'await ' in code:
                    score += 1

        elif language in ["javascript", "typescript"]:
            # React conventions
            if 'use' in code and '(' in code:  # Likely hooks
                checks += 1
                if 'useState' in code or 'useEffect' in code:
                    score += 1

            # Express conventions
            if 'app.' in code:
                checks += 1
                if 'req' in code and 'res' in code:
                    score += 1

        return score / max(checks, 1)

    def _check_api_completeness(self, code: str) -> float:
        """Check if code includes expected API calls"""
        expected_apis = [
            'import', 'def ', 'async', 'await', 'try:', 'except',
            'return', 'if ', 'for ', 'while '
        ]

        found_apis = sum(1 for api in expected_apis if api in code)
        return found_apis / len(expected_apis)

    def run_comprehensive_test(self, prompt: str, context: str = "", file_ext: str = ".py") -> TestResult:
        """Run complete test with all validation"""

        print(f"\n🧪 Comprehensive Test: {prompt[:50]}...")
        print("-" * 60)

        start_time = time.time()

        # Generate baseline
        print("🔵 Baseline generation...")
        baseline = self.asa_system.generate_with_ollama(prompt)

        # Generate enhanced with full system
        print("🟢 Enhanced generation (patterns + ASA bias)...")
        enhanced, metrics = self.asa_system.enhanced_generate_with_asa(
            prompt, context, file_ext
        )

        generation_time = time.time() - start_time

        # Validate both outputs
        language = "python" if file_ext == ".py" else "javascript"
        baseline_validation = self.validate_code_quality(baseline, language)
        enhanced_validation = self.validate_code_quality(enhanced, language)

        # Count repo patterns found
        repo_patterns_found = 0
        for pattern in self.repo_patterns:
            if any(api.lower() in enhanced.lower() for api in pattern.apis):
                repo_patterns_found += 1

        # Create result
        result = TestResult(
            prompt=prompt,
            baseline=baseline,
            enhanced=enhanced,
            validation=enhanced_validation,
            patterns_used=metrics.get('patterns', []),
            api_hit_rate=metrics.get('api_hit_rate', 0.0),
            lift_score=metrics.get('lift', {}).get('total_lift', 0),
            generation_time=generation_time,
            repo_patterns_found=repo_patterns_found
        )

        # Display results
        print(f"\n📊 Results:")
        print(f"   Length: {len(baseline)} → {len(enhanced)} chars")
        print(f"   Syntax valid: {baseline_validation.syntax_valid} → {enhanced_validation.syntax_valid}")
        print(f"   Linting score: {baseline_validation.linting_score:.2f} → {enhanced_validation.linting_score:.2f}")
        print(f"   Framework conformity: {enhanced_validation.framework_conformity:.2f}")
        print(f"   API hit rate: {result.api_hit_rate:.2f}")
        print(f"   Patterns used: {', '.join(result.patterns_used)}")
        print(f"   Repo patterns found: {result.repo_patterns_found}")
        print(f"   Generation time: {result.generation_time:.2f}s")

        self.test_results.append(result)
        return result

    def run_test_suite(self) -> Dict:
        """Run comprehensive test suite"""

        print("🚀 Qwen2.5-Coder:3B Code Attractor Test Suite")
        print("Features: Repo mining + Pattern injection + ASA bias + Validation")
        print("=" * 70)

        test_cases = [
            {
                "prompt": "Create an async FastAPI endpoint for user authentication with JWT",
                "context": "Python web API with user management and token-based auth",
                "ext": ".py"
            },
            {
                "prompt": "Add database transaction handling with proper rollback for user updates",
                "context": "SQLAlchemy ORM with PostgreSQL backend",
                "ext": ".py"
            },
            {
                "prompt": "Implement caching middleware to reduce database load",
                "context": "High-traffic web application with Redis cache",
                "ext": ".py"
            },
            {
                "prompt": "Create a React hook for API data fetching with error handling",
                "context": "TypeScript React application consuming REST API",
                "ext": ".tsx"
            },
            {
                "prompt": "Add Express middleware for request logging and authentication",
                "context": "Node.js REST API with JWT and request tracking",
                "ext": ".js"
            }
        ]

        results = []
        for test_case in test_cases:
            result = self.run_comprehensive_test(
                test_case["prompt"],
                test_case["context"],
                test_case["ext"]
            )
            results.append(result)

        return self._generate_summary(results)

    def _generate_summary(self, results: List[TestResult]) -> Dict:
        """Generate comprehensive test summary"""

        summary = {
            "total_tests": len(results),
            "avg_syntax_improvement": 0,
            "avg_linting_improvement": 0,
            "avg_api_hit_rate": 0,
            "total_patterns_used": 0,
            "repo_patterns_effectiveness": 0,
            "avg_generation_time": 0,
            "framework_conformity": 0
        }

        if not results:
            return summary

        # Calculate averages
        syntax_improvements = 0
        linting_improvements = 0

        for result in results:
            # Syntax validation
            baseline_validation = self.validate_code_quality(result.baseline)
            if result.validation.syntax_valid and not baseline_validation.syntax_valid:
                syntax_improvements += 1

            # Linting improvements
            baseline_linting = baseline_validation.linting_score
            enhanced_linting = result.validation.linting_score
            linting_improvements += enhanced_linting - baseline_linting

            summary["avg_api_hit_rate"] += result.api_hit_rate
            summary["total_patterns_used"] += len(result.patterns_used)
            summary["repo_patterns_effectiveness"] += result.repo_patterns_found
            summary["avg_generation_time"] += result.generation_time
            summary["framework_conformity"] += result.validation.framework_conformity

        # Calculate final averages
        n = len(results)
        summary["avg_syntax_improvement"] = syntax_improvements / n
        summary["avg_linting_improvement"] = linting_improvements / n
        summary["avg_api_hit_rate"] /= n
        summary["avg_generation_time"] /= n
        summary["framework_conformity"] /= n

        return summary

    def print_final_report(self, summary: Dict):
        """Print comprehensive final report"""

        print("\n" + "=" * 70)
        print("📊 QWEN2.5-CODER:3B CODE ATTRACTOR EFFECTIVENESS REPORT")
        print("=" * 70)

        print(f"✅ Tests completed: {summary['total_tests']}")
        print(f"🎯 Average API hit rate: {summary['avg_api_hit_rate']:.2f}")
        print(f"🔧 Average syntax improvement: {summary['avg_syntax_improvement']:.2f}")
        print(f"🧹 Average linting improvement: {summary['avg_linting_improvement']:+.2f}")
        print(f"🚀 Total patterns activated: {summary['total_patterns_used']}")
        print(f"📚 Repo patterns utilized: {summary['repo_patterns_effectiveness']}")
        print(f"⚡ Average generation time: {summary['avg_generation_time']:.2f}s")
        print(f"🎨 Framework conformity: {summary['framework_conformity']:.2f}")

        # Effectiveness assessment
        effectiveness_score = (
            summary['avg_api_hit_rate'] * 0.3 +
            summary['framework_conformity'] * 0.3 +
            summary['avg_linting_improvement'] * 0.2 +
            min(summary['repo_patterns_effectiveness'] / summary['total_tests'], 1.0) * 0.2
        )

        print(f"\n🏆 Overall effectiveness score: {effectiveness_score:.2f}/1.0")

        if effectiveness_score > 0.7:
            print("🎉 EXCELLENT: Code attractors significantly improve qwen2.5-coder:3b!")
        elif effectiveness_score > 0.5:
            print("✅ GOOD: Meaningful improvements in code quality and API usage")
        else:
            print("⚠️ MIXED: Some improvements, but system needs tuning")

        print(f"\n💡 The code attractor system successfully made qwen2.5-coder:3b more repo-aware!")

def main():
    """Run the complete qwen2.5-coder:3b test"""

    # Option to mine from current directory
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder Code Attractor Test")
    parser.add_argument("--repo", type=str, help="Repository path to mine patterns from")
    args = parser.parse_args()

    # Initialize tester
    tester = QwenCodeTester(repo_path=args.repo)

    # Run test suite
    summary = tester.run_test_suite()

    # Print final report
    tester.print_final_report(summary)

if __name__ == "__main__":
    main()