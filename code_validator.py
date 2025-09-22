#!/usr/bin/env python3
"""
Code Validation System for Generated Code
Real linting, testing, and static analysis for code quality metrics
"""

import os
import ast
import re
import subprocess
import tempfile
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LintingResult:
    """Results from code linting"""
    score: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    violations: int = 0
    total_lines: int = 0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class StaticAnalysisResult:
    """Results from static analysis"""
    complexity: int = 0
    security_issues: List[str] = None
    maintainability: float = 0.0
    type_coverage: float = 0.0

    def __post_init__(self):
        if self.security_issues is None:
            self.security_issues = []

@dataclass
class TestResult:
    """Results from test execution"""
    passed: bool = False
    coverage: float = 0.0
    execution_time: float = 0.0
    output: str = ""

class CodeValidator:
    """Comprehensive code validation system"""

    def __init__(self):
        self.temp_dir = None

    def validate_python_code(self, code: str, run_tests: bool = False) -> Dict:
        """Comprehensive Python code validation"""

        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            file_path = Path(temp_dir) / "generated_code.py"

            # Write code to file
            with open(file_path, 'w') as f:
                f.write(code)

            results = {
                "syntax": self._check_python_syntax(code),
                "linting": self._run_python_linting(file_path),
                "static_analysis": self._run_python_static_analysis(file_path),
                "security": self._check_python_security(file_path),
                "imports": self._analyze_python_imports(code),
                "complexity": self._calculate_complexity(code)
            }

            if run_tests:
                results["tests"] = self._run_python_tests(file_path)

            return results

    def validate_javascript_code(self, code: str, is_typescript: bool = False) -> Dict:
        """Comprehensive JavaScript/TypeScript validation"""

        ext = ".ts" if is_typescript else ".js"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / f"generated_code{ext}"

            with open(file_path, 'w') as f:
                f.write(code)

            results = {
                "syntax": self._check_js_syntax(code, is_typescript),
                "linting": self._run_js_linting(file_path, is_typescript),
                "framework_usage": self._analyze_js_frameworks(code),
                "complexity": self._calculate_js_complexity(code)
            }

            return results

    def _check_python_syntax(self, code: str) -> Dict:
        """Check Python syntax validity"""
        try:
            tree = ast.parse(code)
            return {
                "valid": True,
                "error": None,
                "ast_nodes": len(list(ast.walk(tree)))
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }

    def _run_python_linting(self, file_path: Path) -> LintingResult:
        """Run Python linting with flake8/ruff if available"""

        result = LintingResult()
        code_lines = 0

        try:
            with open(file_path, 'r') as f:
                code_lines = len([line for line in f if line.strip()])

            result.total_lines = code_lines

            # Try ruff first (faster)
            if self._tool_available("ruff"):
                output = self._run_command(["ruff", "check", str(file_path)])
                result = self._parse_ruff_output(output, result)

            # Fall back to flake8
            elif self._tool_available("flake8"):
                output = self._run_command(["flake8", str(file_path)])
                result = self._parse_flake8_output(output, result)

            # Manual linting if no tools available
            else:
                result = self._manual_python_linting(file_path)

        except Exception as e:
            result.errors.append(f"Linting failed: {e}")

        # Calculate score
        if result.total_lines > 0:
            violations_per_line = result.violations / result.total_lines
            result.score = max(0.0, 1.0 - violations_per_line)
        else:
            result.score = 1.0

        return result

    def _run_python_static_analysis(self, file_path: Path) -> StaticAnalysisResult:
        """Run static analysis on Python code"""

        result = StaticAnalysisResult()

        try:
            with open(file_path, 'r') as f:
                code = f.read()

            # Calculate cyclomatic complexity
            result.complexity = self._calculate_cyclomatic_complexity(code)

            # Basic maintainability metrics
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if non_empty_lines:
                # Simple maintainability heuristics
                avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
                function_count = code.count('def ')
                class_count = code.count('class ')

                # Normalize to 0-1 scale
                result.maintainability = min(1.0, max(0.0,
                    1.0 - (avg_line_length / 120) * 0.3 +  # Penalty for long lines
                    (function_count + class_count) / len(non_empty_lines) * 0.3  # Bonus for modularity
                ))

            # Type coverage (basic check for type hints)
            type_hints = code.count(': ') + code.count('->') + code.count('typing.')
            total_definitions = code.count('def ') + code.count('class ')
            if total_definitions > 0:
                result.type_coverage = min(1.0, type_hints / total_definitions)

        except Exception as e:
            result.security_issues.append(f"Analysis failed: {e}")

        return result

    def _check_python_security(self, file_path: Path) -> List[str]:
        """Basic Python security checks"""

        security_issues = []

        try:
            with open(file_path, 'r') as f:
                code = f.read()

            # Basic security pattern checks
            if 'eval(' in code:
                security_issues.append("Use of eval() detected - potential code injection risk")

            if 'exec(' in code:
                security_issues.append("Use of exec() detected - potential code injection risk")

            if 'subprocess.call' in code and 'shell=True' in code:
                security_issues.append("shell=True in subprocess call - potential command injection")

            if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
                security_issues.append("Hardcoded password detected")

            if 'pickle.loads' in code:
                security_issues.append("pickle.loads() usage - potential deserialization vulnerability")

        except Exception as e:
            security_issues.append(f"Security check failed: {e}")

        return security_issues

    def _analyze_python_imports(self, code: str) -> Dict:
        """Analyze Python import patterns"""

        try:
            tree = ast.parse(code)
            imports = {
                "stdlib": [],
                "third_party": [],
                "local": [],
                "star_imports": 0
            }

            stdlib_modules = {
                'os', 'sys', 'json', 'time', 'datetime', 're', 'collections',
                'functools', 'itertools', 'pathlib', 'typing', 'asyncio'
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module in stdlib_modules:
                            imports["stdlib"].append(alias.name)
                        else:
                            imports["third_party"].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module in stdlib_modules:
                            imports["stdlib"].append(node.module)
                        else:
                            imports["third_party"].append(node.module)

                    # Check for star imports
                    for alias in node.names:
                        if alias.name == '*':
                            imports["star_imports"] += 1

            return imports

        except Exception:
            return {"error": "Import analysis failed"}

    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity metrics"""

        try:
            # Simple complexity based on control flow
            complexity = 1  # Base complexity

            # Add complexity for control structures
            complexity += code.count('if ')
            complexity += code.count('elif ')
            complexity += code.count('for ')
            complexity += code.count('while ')
            complexity += code.count('except ')
            complexity += code.count('and ')
            complexity += code.count('or ')

            return complexity

        except Exception:
            return 0

    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""

        try:
            tree = ast.parse(code)
            complexity = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return max(1, complexity)

        except Exception:
            return 1

    def _check_js_syntax(self, code: str, is_typescript: bool = False) -> Dict:
        """Basic JavaScript/TypeScript syntax check"""

        # Simple bracket/parentheses matching
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []

        try:
            for char in code:
                if char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return {"valid": False, "error": f"Unmatched closing bracket: {char}"}
                    if brackets[stack[-1]] != char:
                        return {"valid": False, "error": f"Mismatched brackets"}
                    stack.pop()

            if stack:
                return {"valid": False, "error": f"Unclosed brackets: {stack}"}

            return {"valid": True, "error": None}

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _run_js_linting(self, file_path: Path, is_typescript: bool = False) -> LintingResult:
        """Run JavaScript/TypeScript linting"""

        result = LintingResult()

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            result.total_lines = len([line for line in lines if line.strip()])

            # Manual JS linting rules
            violations = 0
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # Basic linting rules
                if line.endswith(';') is False and not line.endswith('{') and not line.endswith('}'):
                    if not any(keyword in line for keyword in ['if', 'for', 'while', 'function', 'const', 'let', 'var']):
                        violations += 1
                        result.warnings.append(f"Line {i}: Missing semicolon")

                if 'var ' in line:
                    violations += 1
                    result.warnings.append(f"Line {i}: Use let/const instead of var")

                if '==' in line and '===' not in line:
                    violations += 1
                    result.warnings.append(f"Line {i}: Use === instead of ==")

            result.violations = violations

            if result.total_lines > 0:
                result.score = max(0.0, 1.0 - (violations / result.total_lines))
            else:
                result.score = 1.0

        except Exception as e:
            result.errors.append(f"JS linting failed: {e}")

        return result

    def _analyze_js_frameworks(self, code: str) -> Dict:
        """Analyze JavaScript framework usage"""

        frameworks = {
            "react": 0,
            "express": 0,
            "node": 0,
            "async": 0
        }

        # React patterns
        if re.search(r'use[A-Z]\w*\s*\(', code):
            frameworks["react"] += 1

        if 'useState' in code or 'useEffect' in code:
            frameworks["react"] += 1

        # Express patterns
        if 'app.get(' in code or 'app.post(' in code:
            frameworks["express"] += 1

        if 'req.' in code and 'res.' in code:
            frameworks["express"] += 1

        # Async patterns
        if 'async ' in code and 'await ' in code:
            frameworks["async"] += 1

        return frameworks

    def _calculate_js_complexity(self, code: str) -> int:
        """Calculate JavaScript complexity"""

        complexity = 1

        # Control structures
        complexity += len(re.findall(r'\bif\b', code))
        complexity += len(re.findall(r'\belse\b', code))
        complexity += len(re.findall(r'\bfor\b', code))
        complexity += len(re.findall(r'\bwhile\b', code))
        complexity += len(re.findall(r'\btry\b', code))
        complexity += len(re.findall(r'\bcatch\b', code))

        return complexity

    def _manual_python_linting(self, file_path: Path) -> LintingResult:
        """Manual Python linting when tools unavailable"""

        result = LintingResult()

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            result.total_lines = len([line for line in lines if line.strip()])
            violations = 0

            for i, line in enumerate(lines, 1):
                line_strip = line.strip()
                if not line_strip:
                    continue

                # Basic PEP 8 checks
                if len(line.rstrip()) > 120:
                    violations += 1
                    result.warnings.append(f"Line {i}: Line too long ({len(line.rstrip())} > 120)")

                if line_strip.endswith('except:'):
                    violations += 1
                    result.errors.append(f"Line {i}: Bare except clause")

                if 'import *' in line:
                    violations += 1
                    result.warnings.append(f"Line {i}: Star import")

                # Indentation check (basic)
                if line.startswith('    ') is False and line.startswith('\t') is False and line_strip and not line_strip.startswith('#'):
                    if any(line_strip.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:']):
                        pass  # Top-level is OK
                    else:
                        # Check if this should be indented
                        prev_line = lines[i-2].strip() if i > 1 else ''
                        if prev_line.endswith(':'):
                            violations += 1
                            result.errors.append(f"Line {i}: Expected indentation")

            result.violations = violations

        except Exception as e:
            result.errors.append(f"Manual linting failed: {e}")

        return result

    def _parse_ruff_output(self, output: str, result: LintingResult) -> LintingResult:
        """Parse ruff linting output"""

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            result.violations += 1

            if ' E ' in line or ' F ' in line:  # Errors
                result.errors.append(line)
            else:  # Warnings
                result.warnings.append(line)

        return result

    def _parse_flake8_output(self, output: str, result: LintingResult) -> LintingResult:
        """Parse flake8 linting output"""

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            result.violations += 1
            result.warnings.append(line)

        return result

    def _tool_available(self, tool: str) -> bool:
        """Check if a command-line tool is available"""
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_command(self, cmd: List[str]) -> str:
        """Run command and return output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Command failed: {e}"

def main():
    """Demo the code validator"""

    print("🔍 Code Validator Demo")
    print("=" * 50)

    validator = CodeValidator()

    # Test Python code
    python_code = '''
import asyncio
from fastapi import FastAPI, Depends

app = FastAPI()

async def get_db():
    try:
        db = connect_to_db()
        yield db
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        db.close()

@app.get("/users/{user_id}")
async def get_user(user_id: int, db=Depends(get_db)):
    try:
        user = await db.fetch_user(user_id)
        return {"user": user}
    except Exception:
        raise HTTPException(status_code=404)
'''

    print("🐍 Python validation:")
    py_results = validator.validate_python_code(python_code)

    print(f"   Syntax valid: {py_results['syntax']['valid']}")
    print(f"   Linting score: {py_results['linting'].score:.2f}")
    print(f"   Complexity: {py_results['complexity']}")
    print(f"   Security issues: {len(py_results['security'])}")

    # Test JavaScript code
    js_code = '''
const express = require('express');
const app = express();

app.use(express.json());

const authMiddleware = (req, res, next) => {
    const token = req.headers.authorization;
    if (!token) {
        return res.status(401).json({ error: 'No token' });
    }
    next();
};

app.get('/api/users', authMiddleware, async (req, res) => {
    try {
        const users = await User.findAll();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
'''

    print("\n📜 JavaScript validation:")
    js_results = validator.validate_javascript_code(js_code)

    print(f"   Syntax valid: {js_results['syntax']['valid']}")
    print(f"   Linting score: {js_results['linting'].score:.2f}")
    print(f"   Complexity: {js_results['complexity']}")
    print(f"   Framework usage: {js_results['framework_usage']}")

if __name__ == "__main__":
    main()