#!/usr/bin/env python3
"""
Repo Mining System for Code Attractors
Automatically extract patterns from codebases to generate repo-specific attractors
"""

import ast
import os
import re
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from code_attractor_system import CodePattern

@dataclass
class PatternStats:
    """Statistics for discovered patterns"""
    frequency: int = 0
    files: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    resonance: float = 0.0

class RepoMiner:
    """Mine code patterns from repositories"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.patterns = defaultdict(PatternStats)
        self.framework_usage = Counter()
        self.import_patterns = Counter()
        self.error_patterns = Counter()

    def scan_repository(self, max_files: int = 200) -> Dict[str, PatternStats]:
        """Scan repository for code patterns"""

        print(f"🔍 Mining patterns from {self.repo_path}")

        file_count = 0

        # Scan Python files
        for py_file in self.repo_path.rglob("*.py"):
            if file_count >= max_files:
                break
            if self._should_skip_file(py_file):
                continue

            self._analyze_python_file(py_file)
            file_count += 1

        # Scan JavaScript/TypeScript files
        for js_file in self.repo_path.rglob("*.js"):
            if file_count >= max_files:
                break
            if self._should_skip_file(js_file):
                continue

            self._analyze_js_file(js_file)
            file_count += 1

        for ts_file in self.repo_path.rglob("*.ts"):
            if file_count >= max_files:
                break
            if self._should_skip_file(ts_file):
                continue

            self._analyze_ts_file(ts_file)
            file_count += 1

        # Calculate resonance scores
        self._calculate_resonance()

        print(f"📊 Analyzed {file_count} files, found {len(self.patterns)} patterns")

        return dict(self.patterns)

    def _should_skip_file(self, file_path: Path) -> bool:
        """Skip certain files/directories"""
        skip_dirs = {
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'dist', 'build', '.venv', 'venv', 'env'
        }

        skip_files = {
            'test_', '_test', '.test.', '.spec.', '__init__.py'
        }

        # Skip if in skip directory
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            return True

        # Skip test files
        if any(skip_pattern in file_path.name for skip_pattern in skip_files):
            return True

        return False

    def _analyze_python_file(self, file_path: Path):
        """Analyze Python file for patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return

            # Extract patterns
            self._extract_python_patterns(tree, file_path, content)

        except Exception as e:
            # Skip files with encoding issues
            pass

    def _extract_python_patterns(self, tree: ast.AST, file_path: Path, content: str):
        """Extract patterns from Python AST"""

        # Track imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_patterns[alias.name] += 1
                    self._record_framework_usage(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.import_patterns[node.module] += 1
                    self._record_framework_usage(node.module)

            # Async function patterns
            elif isinstance(node, ast.AsyncFunctionDef):
                pattern_name = "async_function"
                self._record_pattern(pattern_name, file_path, f"async def {node.name}")

                # Check for common async patterns
                if any(isinstance(n, ast.Await) for n in ast.walk(node)):
                    self._record_pattern("async_await", file_path, f"async def {node.name} with await")

            # Error handling patterns
            elif isinstance(node, ast.Try):
                self._analyze_error_handling(node, file_path)

            # Decorator patterns
            elif isinstance(node, ast.FunctionDef) and node.decorator_list:
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        self._record_pattern(f"decorator_{decorator.id}", file_path, f"@{decorator.id}")
                    elif isinstance(decorator, ast.Attribute):
                        self._record_pattern(f"decorator_{decorator.attr}", file_path, f"@{decorator.attr}")

        # Text-based pattern detection
        self._detect_text_patterns(content, file_path)

    def _analyze_js_file(self, file_path: Path):
        """Analyze JavaScript file for patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self._detect_js_patterns(content, file_path)

        except Exception:
            pass

    def _analyze_ts_file(self, file_path: Path):
        """Analyze TypeScript file for patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self._detect_js_patterns(content, file_path)  # Same patterns as JS
            self._detect_typescript_patterns(content, file_path)

        except Exception:
            pass

    def _detect_js_patterns(self, content: str, file_path: Path):
        """Detect JavaScript patterns"""

        # React hooks
        hooks = re.findall(r'(use[A-Z]\w*)\s*\(', content)
        for hook in hooks:
            self._record_pattern(f"react_{hook.lower()}", file_path, f"{hook}()")

        # Express patterns
        if 'express' in content.lower():
            if re.search(r'app\.(get|post|put|delete)', content):
                self._record_pattern("express_route", file_path, "app.get/post/put/delete")
            if 'middleware' in content.lower():
                self._record_pattern("express_middleware", file_path, "middleware")

        # Async/await
        if re.search(r'async\s+\w+|await\s+', content):
            self._record_pattern("js_async_await", file_path, "async/await")

        # Promise patterns
        if '.then(' in content or '.catch(' in content:
            self._record_pattern("promise_chain", file_path, ".then/.catch")

    def _detect_typescript_patterns(self, content: str, file_path: Path):
        """Detect TypeScript-specific patterns"""

        # Interface definitions
        interfaces = re.findall(r'interface\s+(\w+)', content)
        for interface in interfaces:
            self._record_pattern("typescript_interface", file_path, f"interface {interface}")

        # Type definitions
        types = re.findall(r'type\s+(\w+)\s*=', content)
        for type_def in types:
            self._record_pattern("typescript_type", file_path, f"type {type_def}")

    def _detect_text_patterns(self, content: str, file_path: Path):
        """Detect patterns using regex"""

        # FastAPI patterns
        if 'fastapi' in content.lower():
            if '@app.' in content:
                self._record_pattern("fastapi_route", file_path, "@app.get/@app.post")
            if 'Depends(' in content:
                self._record_pattern("fastapi_dependency", file_path, "Depends()")
            if 'HTTPException' in content:
                self._record_pattern("fastapi_exception", file_path, "HTTPException")

        # Database patterns
        if any(db in content.lower() for db in ['sqlalchemy', 'asyncpg', 'psycopg']):
            if 'session.' in content:
                self._record_pattern("db_session", file_path, "session operations")
            if 'commit()' in content:
                self._record_pattern("db_transaction", file_path, "commit/rollback")

        # Caching patterns
        if any(cache in content.lower() for cache in ['redis', 'cache', 'lru_cache']):
            self._record_pattern("caching", file_path, "caching implementation")

    def _analyze_error_handling(self, try_node: ast.Try, file_path: Path):
        """Analyze try/except patterns"""

        exception_types = []
        for handler in try_node.handlers:
            if handler.type:
                if isinstance(handler.type, ast.Name):
                    exception_types.append(handler.type.id)
                elif isinstance(handler.type, ast.Attribute):
                    exception_types.append(handler.type.attr)

        if exception_types:
            self._record_pattern("error_handling", file_path, f"try/except {'/'.join(exception_types)}")
        else:
            self._record_pattern("bare_except", file_path, "bare except (antipattern)")

    def _record_framework_usage(self, import_name: str):
        """Record framework usage for resonance calculation"""
        frameworks = {
            'fastapi': ['fastapi'],
            'express': ['express'],
            'react': ['react'],
            'sqlalchemy': ['sqlalchemy'],
            'asyncio': ['asyncio'],
            'redis': ['redis'],
            'pytest': ['pytest'],
            'numpy': ['numpy'],
            'pandas': ['pandas']
        }

        for framework, modules in frameworks.items():
            if any(module in import_name.lower() for module in modules):
                self.framework_usage[framework] += 1

    def _record_pattern(self, pattern_name: str, file_path: Path, example: str):
        """Record a discovered pattern"""
        stats = self.patterns[pattern_name]
        stats.frequency += 1
        stats.files.add(str(file_path))

        if len(stats.examples) < 5:  # Keep up to 5 examples
            stats.examples.append(example)

    def _calculate_resonance(self):
        """Calculate resonance scores for patterns"""
        max_frequency = max((stats.frequency for stats in self.patterns.values()), default=1)

        for pattern_name, stats in self.patterns.items():
            # Base resonance on frequency and file spread
            freq_score = stats.frequency / max_frequency
            file_spread = min(len(stats.files) / 10, 1.0)  # Bonus for appearing in multiple files

            # Penalty for likely antipatterns
            penalty = 0
            if 'bare_except' in pattern_name:
                penalty = 0.3

            stats.resonance = min(0.95, max(0.1, (freq_score * 0.7 + file_spread * 0.3) - penalty))

    def generate_code_attractors(self, min_frequency: int = 2, min_resonance: float = 0.3) -> List[CodePattern]:
        """Generate CodePattern objects from discovered patterns"""

        attractors = []

        for pattern_name, stats in self.patterns.items():
            if stats.frequency < min_frequency or stats.resonance < min_resonance:
                continue

            # Map pattern names to meaningful summaries
            summary = self._generate_pattern_summary(pattern_name, stats)
            apis = self._extract_apis_from_pattern(pattern_name, stats)

            # Generate snippet from examples
            snippet = stats.examples[0] if stats.examples else ""

            # Identify antipatterns
            antipatterns = []
            if 'bare_except' in pattern_name:
                antipatterns = ["bare except clause", "missing specific exception handling"]

            attractor = CodePattern(
                name=pattern_name,
                summary=summary,
                apis=apis,
                snippets=[snippet] if snippet else [],
                antipatterns=antipatterns,
                resonance=stats.resonance,
                source="repo_mined"
            )

            attractors.append(attractor)

        return sorted(attractors, key=lambda x: x.resonance, reverse=True)

    def _generate_pattern_summary(self, pattern_name: str, stats: PatternStats) -> str:
        """Generate human-readable summary for pattern"""

        summaries = {
            'async_function': 'Async function definitions for concurrent operations',
            'async_await': 'Async/await pattern for non-blocking operations',
            'fastapi_route': 'FastAPI route handler with decorators',
            'fastapi_dependency': 'FastAPI dependency injection pattern',
            'express_route': 'Express.js route handler pattern',
            'react_usestate': 'React useState hook for state management',
            'react_useeffect': 'React useEffect hook for side effects',
            'error_handling': 'Structured exception handling with specific types',
            'db_transaction': 'Database transaction with commit/rollback',
            'caching': 'Caching implementation for performance optimization'
        }

        return summaries.get(pattern_name, f"Pattern: {pattern_name.replace('_', ' ')}")

    def _extract_apis_from_pattern(self, pattern_name: str, stats: PatternStats) -> List[str]:
        """Extract relevant API names from pattern"""

        api_mappings = {
            'fastapi_route': ['@app.get', '@app.post', 'APIRouter', 'Depends'],
            'fastapi_dependency': ['Depends', 'HTTPException', 'status'],
            'express_route': ['app.get', 'app.post', 'router', 'middleware'],
            'react_usestate': ['useState', 'setState'],
            'react_useeffect': ['useEffect', 'useCallback', 'dependency array'],
            'async_await': ['async', 'await', 'asyncio'],
            'db_transaction': ['session', 'commit', 'rollback', 'transaction'],
            'caching': ['cache', 'redis', 'lru_cache', 'ttl']
        }

        return api_mappings.get(pattern_name, [])

    def export_patterns(self, output_file: str):
        """Export discovered patterns to JSON"""
        export_data = {
            'repo_path': str(self.repo_path),
            'framework_usage': dict(self.framework_usage),
            'patterns': {
                name: {
                    'frequency': stats.frequency,
                    'files': list(stats.files),
                    'examples': stats.examples,
                    'resonance': stats.resonance
                }
                for name, stats in self.patterns.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"💾 Exported patterns to {output_file}")

def main():
    """Demo repo mining"""

    # Mine current directory as example
    repo_path = "."

    print("🔍 Repo Mining Demo")
    print(f"Mining patterns from: {repo_path}")
    print("=" * 50)

    miner = RepoMiner(repo_path)
    patterns = miner.scan_repository(max_files=50)

    # Show top patterns
    print("\n📊 Top Discovered Patterns:")
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1].resonance, reverse=True)

    for i, (name, stats) in enumerate(sorted_patterns[:10], 1):
        print(f"   {i:2}. {name:20} freq={stats.frequency:2} resonance={stats.resonance:.2f}")
        if stats.examples:
            print(f"       example: {stats.examples[0][:60]}...")

    # Generate attractors
    attractors = miner.generate_code_attractors(min_frequency=1, min_resonance=0.1)

    print(f"\n🎯 Generated {len(attractors)} code attractors")
    for attractor in attractors[:5]:
        print(f"   • {attractor.name}: {attractor.summary[:50]}... (r={attractor.resonance:.2f})")

    # Export patterns
    miner.export_patterns("discovered_patterns.json")

    print(f"\n🚀 Framework usage detected: {dict(miner.framework_usage)}")

if __name__ == "__main__":
    main()