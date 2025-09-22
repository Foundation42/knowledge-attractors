#!/usr/bin/env python3
"""
Smoke Tests for Code Attractor System
Quick validation that core components work
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

def test_repo_mining():
    """Test basic repo mining functionality"""
    from repo_mining import RepoMiner

    # Create a temporary repo with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_code.py"
        test_file.write_text("""
import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
async def test_endpoint():
    try:
        result = await some_async_operation()
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500)
""")

        # Mine patterns
        miner = RepoMiner(temp_dir)
        patterns = miner.scan_repository(max_files=10)

        # Should find some patterns
        assert len(patterns) > 0

        # Should detect FastAPI patterns
        pattern_names = list(patterns.keys())
        assert any('fastapi' in name for name in pattern_names)

def test_code_attractor_system():
    """Test code attractor pattern selection"""
    from code_attractor_system import CodeAttractorSystem

    system = CodeAttractorSystem()

    # Should have curated patterns
    assert len(system.patterns) > 0

    # Test pattern selection
    context = "Create FastAPI endpoint with async database calls"
    patterns = system.select_patterns_for_context(context, ".py")

    # Should select relevant patterns
    assert len(patterns) > 0

    # Test compact serialization
    compact_json = system.build_consider_code("test_theme", patterns[:3])
    assert len(compact_json) > 0
    assert len(compact_json.encode('utf-8')) < 2000  # Reasonable size

def test_asa_bias_system():
    """Test ASA bias framework detection"""
    from asa_bias_system import ASABiasSystem

    system = ASABiasSystem()

    # Test framework detection
    prompt = "Create async FastAPI endpoint with Redis caching"
    frameworks = system.detect_framework_context(prompt)

    assert 'fastapi' in frameworks or 'asyncio' in frameworks

    # Test API seed generation
    patterns = system.select_patterns_for_context(prompt, ".py")
    api_seed = system.create_api_seed_prefix(frameworks, patterns)

    assert len(api_seed) > 0

def test_code_validator():
    """Test code validation functionality"""
    from code_validator import CodeValidator

    validator = CodeValidator()

    # Test valid Python code
    valid_code = """
import asyncio

async def test_function():
    try:
        result = await asyncio.sleep(1)
        return result
    except Exception as e:
        print(f"Error: {e}")
"""

    results = validator.validate_python_code(valid_code)

    assert results['syntax']['valid'] is True
    assert results['linting'].score >= 0.0

    # Test invalid Python code
    invalid_code = "def broken_function(\n    pass"

    results = validator.validate_python_code(invalid_code)
    assert results['syntax']['valid'] is False

def test_compact_serialization():
    """Test that compact serialization stays under size limits"""
    from code_attractor_system import CodeAttractorSystem

    system = CodeAttractorSystem()

    # Create test patterns
    patterns = list(system.patterns.values())[:3]

    # Test compact serialization
    compact_json = system.build_consider_code("test_theme", patterns)

    # Check size constraints
    full_block = f"<consider>\n{compact_json}\n</consider>"
    size_bytes = len(full_block.encode('utf-8'))

    # Should be under reasonable limit for most cases
    assert size_bytes < 2000, f"Block too large: {size_bytes} bytes"

def test_cli_functionality():
    """Test CLI module basic functionality"""
    from cli_module import check_command

    # Should not crash
    result = check_command()
    assert result in [0, 1]  # Valid exit codes

def test_pattern_resonance_calculation():
    """Test pattern resonance scoring"""
    from repo_mining import RepoMiner

    # Create test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files with repeated patterns
        for i in range(3):
            test_file = Path(temp_dir) / f"test_{i}.py"
            test_file.write_text("""
try:
    result = some_operation()
except Exception as e:
    handle_error(e)
""")

        miner = RepoMiner(temp_dir)
        patterns = miner.scan_repository()

        # Should calculate reasonable resonance scores
        for pattern_name, stats in patterns.items():
            assert 0.0 <= stats.resonance <= 1.0

def test_framework_token_vocabulary():
    """Test framework token detection"""
    from asa_bias_system import ASABiasSystem

    system = ASABiasSystem()

    # Should have framework tokens
    assert len(system.framework_tokens) > 0

    # Should include common frameworks
    tokens_lower = {token.lower() for token in system.framework_tokens}
    assert 'fastapi' in tokens_lower
    assert 'async' in tokens_lower
    assert 'react' in tokens_lower

def test_configuration_loading():
    """Test configuration file handling"""

    # Test default config
    config_path = "code_attractor_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        assert 'system' in config
        assert 'features' in config
        assert 'limits' in config

def test_antipattern_detection():
    """Test antipattern identification"""
    from repo_mining import RepoMiner

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file with antipatterns
        test_file = Path(temp_dir) / "bad_code.py"
        test_file.write_text("""
try:
    risky_operation()
except:  # Bare except - antipattern
    pass

eval("some_code")  # Security issue
""")

        miner = RepoMiner(temp_dir)
        patterns = miner.scan_repository()

        # Should detect bare except as antipattern
        assert 'bare_except' in patterns

# Evaluation framework for repo-specific tasks
class CodegenEvaluator:
    """Evaluation framework for code generation tasks"""

    def __init__(self):
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self):
        """Load repo-specific test cases"""
        return [
            {
                "prompt": "Create async FastAPI endpoint for user authentication",
                "expected_apis": ["@app.post", "async def", "Depends", "HTTPException"],
                "repo_context": "Python web API with FastAPI framework",
                "language": "python"
            },
            {
                "prompt": "Add Redis caching to database query function",
                "expected_apis": ["redis", "cache", "async with", "ttl"],
                "repo_context": "High-performance web application",
                "language": "python"
            },
            {
                "prompt": "Create React hook for API data fetching",
                "expected_apis": ["useState", "useEffect", "fetch", "loading"],
                "repo_context": "React TypeScript frontend",
                "language": "typescript"
            }
        ]

    def evaluate_pattern_hit_rate(self, enhanced_code: str, test_case: dict) -> float:
        """Calculate pattern hit rate for generated code"""
        code_lower = enhanced_code.lower()
        expected_apis = test_case["expected_apis"]

        hits = sum(1 for api in expected_apis if api.lower() in code_lower)
        hit_rate = hits / len(expected_apis)

        return hit_rate

    def run_evaluation(self, code_generator_func) -> dict:
        """Run full evaluation suite"""
        results = []

        for test_case in self.test_cases:
            try:
                # Generate code
                enhanced_code = code_generator_func(
                    test_case["prompt"],
                    test_case["repo_context"]
                )

                # Calculate metrics
                hit_rate = self.evaluate_pattern_hit_rate(enhanced_code, test_case)

                results.append({
                    "prompt": test_case["prompt"],
                    "hit_rate": hit_rate,
                    "success": hit_rate >= 0.5  # 50% of expected APIs
                })

            except Exception as e:
                results.append({
                    "prompt": test_case["prompt"],
                    "hit_rate": 0.0,
                    "success": False,
                    "error": str(e)
                })

        # Calculate summary
        successful = sum(1 for r in results if r["success"])
        avg_hit_rate = sum(r["hit_rate"] for r in results) / len(results)

        return {
            "total_tests": len(results),
            "successful": successful,
            "success_rate": successful / len(results),
            "average_hit_rate": avg_hit_rate,
            "results": results
        }

def test_evaluation_framework():
    """Test the evaluation framework"""
    evaluator = CodegenEvaluator()

    # Mock code generator
    def mock_generator(prompt, context):
        if "fastapi" in prompt.lower():
            return "@app.post('/auth')\nasync def authenticate(user: User = Depends(get_user)):\n    raise HTTPException(status_code=401)"
        elif "redis" in prompt.lower():
            return "async with redis.pipeline() as pipe:\n    result = await cache.get(key, ttl=300)"
        elif "react" in prompt.lower():
            return "const [data, setData] = useState(null);\nuseEffect(() => {\n    fetch('/api').then(setData);\n}, []);"
        else:
            return "// Generic code"

    # Run evaluation
    results = evaluator.run_evaluation(mock_generator)

    assert results["total_tests"] > 0
    assert 0.0 <= results["success_rate"] <= 1.0
    assert 0.0 <= results["average_hit_rate"] <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])