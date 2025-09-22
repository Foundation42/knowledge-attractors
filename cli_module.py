#!/usr/bin/env python3
"""
Code Attractor CLI Module
Command-line interface for CI/CD integration
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

def check_command():
    """Run code attractor checks for CI/CD"""

    print("🔍 Running Code Attractor checks...")

    # Check if we have required files
    required_files = ['code_attractor_system.py', 'repo_mining.py', 'code_validator.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return 1

    # Run basic validation
    try:
        from repo_mining import RepoMiner
        from code_validator import CodeValidator

        # Quick pattern check
        miner = RepoMiner(".")
        patterns = miner.scan_repository(max_files=20)

        if len(patterns) == 0:
            print("⚠️  No patterns found in repository")
        else:
            print(f"✅ Found {len(patterns)} code patterns")

        # Check configuration
        if os.path.exists('code_attractor_config.json'):
            with open('code_attractor_config.json') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded: {config.get('version', 'unknown')}")
        else:
            print("⚠️  No configuration file found (code_attractor_config.json)")

        print("✅ Code Attractor checks passed")
        return 0

    except Exception as e:
        print(f"❌ Code Attractor check failed: {e}")
        return 1

def mine_command(args):
    """Mine patterns and emit to file"""

    print(f"🔍 Mining patterns from {args.repo_path}")

    try:
        from repo_mining import RepoMiner

        miner = RepoMiner(args.repo_path)
        patterns = miner.scan_repository(max_files=args.max_files)

        # Export patterns
        if args.emit:
            miner.export_patterns(args.emit)
            print(f"💾 Patterns exported to {args.emit}")

        # Generate summary
        summary = {
            "total_patterns": len(patterns),
            "high_resonance": len([p for p in patterns.values() if p.resonance > 0.7]),
            "framework_usage": dict(miner.framework_usage),
            "top_patterns": [
                {
                    "name": name,
                    "frequency": stats.frequency,
                    "resonance": stats.resonance
                }
                for name, stats in sorted(patterns.items(),
                                        key=lambda x: x[1].resonance,
                                        reverse=True)[:5]
            ]
        }

        print(f"📊 Mining complete: {summary['total_patterns']} patterns found")
        return 0

    except Exception as e:
        print(f"❌ Pattern mining failed: {e}")
        return 1

def validate_command(args):
    """Validate code quality"""

    print("🔍 Running code validation...")

    try:
        from code_validator import CodeValidator

        validator = CodeValidator()

        # Find code files to validate
        code_files = []
        for ext in ['.py', '.js', '.ts', '.tsx']:
            code_files.extend(Path('.').rglob(f'*{ext}'))

        if args.files:
            # Validate specific files
            code_files = [Path(f) for f in args.files if Path(f).exists()]

        total_score = 0
        file_count = 0

        for file_path in code_files[:20]:  # Limit for CI
            if file_path.name.startswith('.'):
                continue

            try:
                with open(file_path, 'r') as f:
                    code = f.read()

                if file_path.suffix == '.py':
                    results = validator.validate_python_code(code)
                    score = results['linting'].score
                else:
                    results = validator.validate_javascript_code(code,
                                                               file_path.suffix == '.ts')
                    score = results['linting'].score

                total_score += score
                file_count += 1

            except Exception as e:
                print(f"⚠️  Skipped {file_path}: {e}")

        avg_score = total_score / max(file_count, 1)

        report = {
            "files_validated": file_count,
            "average_score": avg_score,
            "overall_score": avg_score,
            "threshold_passed": avg_score >= 0.7
        }

        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)

        print(f"📊 Validation complete: {avg_score:.2f} average score")

        if args.check and not report["threshold_passed"]:
            print("❌ Code quality below threshold (0.7)")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Code validation failed: {e}")
        return 1

def test_command(args):
    """Test with qwen2.5-coder"""

    print("🧪 Testing with qwen2.5-coder...")

    try:
        from qwen_code_test import QwenCodeTester

        tester = QwenCodeTester(repo_path=args.repo)

        if args.mode == 'ci':
            # Quick CI test
            test_cases = [
                {
                    "prompt": "Create a simple FastAPI endpoint",
                    "context": "Python web API",
                    "ext": ".py"
                },
                {
                    "prompt": "Add error handling to async function",
                    "context": "Async Python code",
                    "ext": ".py"
                }
            ]

            results = []
            for test_case in test_cases:
                try:
                    result = tester.run_comprehensive_test(
                        test_case["prompt"],
                        test_case["context"],
                        test_case["ext"]
                    )
                    results.append(result)
                except Exception as e:
                    print(f"⚠️  Test skipped: {e}")

            if results:
                summary = tester._generate_summary(results)

                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(summary, f, indent=2)

                print(f"✅ Qwen test complete: effectiveness {summary.get('framework_conformity', 0):.2f}")
            else:
                print("⚠️  No tests completed (Ollama may not be available)")

        return 0

    except Exception as e:
        print(f"❌ Qwen test failed: {e}")
        return 1

def main():
    """Main CLI entry point"""

    parser = argparse.ArgumentParser(description="Code Attractor System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Check command
    check_parser = subparsers.add_parser('check', help='Run CI/CD checks')

    # Mine command
    mine_parser = subparsers.add_parser('mine', help='Mine repository patterns')
    mine_parser.add_argument('repo_path', nargs='?', default='.', help='Repository path')
    mine_parser.add_argument('--emit', help='Output patterns to file')
    mine_parser.add_argument('--max-files', type=int, default=100, help='Max files to scan')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate code quality')
    validate_parser.add_argument('--files', nargs='*', help='Specific files to validate')
    validate_parser.add_argument('--report', help='Output validation report')
    validate_parser.add_argument('--check', action='store_true', help='Exit with error if quality below threshold')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test with qwen2.5-coder')
    test_parser.add_argument('--repo', help='Repository path')
    test_parser.add_argument('--mode', choices=['ci', 'full'], default='full', help='Test mode')
    test_parser.add_argument('--output', help='Output results to file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'check':
        return check_command()
    elif args.command == 'mine':
        return mine_command(args)
    elif args.command == 'validate':
        return validate_command(args)
    elif args.command == 'test':
        return test_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())