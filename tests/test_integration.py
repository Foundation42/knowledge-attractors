#!/usr/bin/env python3
"""
Integration Tests for Code Attractor System
End-to-end testing of the complete pipeline
"""

import pytest
import tempfile
import json
from pathlib import Path

@pytest.mark.integration
def test_complete_pipeline():
    """Test the complete pipeline: mine → inject → validate"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a realistic test repository
        repo_path = Path(temp_dir)

        # Create FastAPI application
        (repo_path / "main.py").write_text("""
from fastapi import FastAPI, Depends, HTTPException
import asyncio

app = FastAPI()

async def get_db():
    # Database connection logic
    pass

@app.get("/users/{user_id}")
async def get_user(user_id: int, db=Depends(get_db)):
    try:
        async with asyncio.timeout(2.0):
            user = await db.fetch_user(user_id)
            return {"user": user}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users")
async def create_user(user_data: dict, db=Depends(get_db)):
    try:
        async with db.transaction():
            user = await db.create_user(user_data)
            return {"user": user}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
""")

        # Create React component
        (repo_path / "frontend.tsx").write_text("""
import React, { useState, useEffect } from 'react';

const UserProfile: React.FC = () => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchUser = async () => {
            try {
                const response = await fetch('/api/user');
                const userData = await response.json();
                setUser(userData);
            } catch (error) {
                console.error('Failed to fetch user:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchUser();
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <div>
            <h1>{user?.name}</h1>
            <p>{user?.email}</p>
        </div>
    );
};

export default UserProfile;
""")

        # Step 1: Mine repository patterns
        from repo_mining import RepoMiner

        miner = RepoMiner(str(repo_path))
        patterns = miner.scan_repository(max_files=10)

        # Should discover patterns from our test files
        assert len(patterns) > 0

        # Should find specific patterns
        pattern_names = list(patterns.keys())
        assert any('async' in name for name in pattern_names)
        assert any('fastapi' in name for name in pattern_names)

        # Step 2: Generate code attractors
        attractors = miner.generate_code_attractors(min_frequency=1, min_resonance=0.1)
        assert len(attractors) > 0

        # Step 3: Test pattern injection
        from asa_bias_system import ASABiasSystem

        system = ASABiasSystem()

        # Add mined patterns to system
        for attractor in attractors:
            system.patterns[attractor.name] = attractor

        # Test code generation with patterns
        prompt = "Create an async FastAPI endpoint with error handling"
        context = "Python web API with database operations"

        # This would normally call Ollama, but we'll test the setup
        patterns_selected = system.select_patterns_for_context(f"{prompt} {context}", ".py")
        frameworks = system.detect_framework_context(prompt, context)

        assert len(patterns_selected) > 0
        assert 'fastapi' in frameworks or 'asyncio' in frameworks

        # Step 4: Test validation
        from code_validator import CodeValidator

        validator = CodeValidator()

        # Read our test code
        test_code = (repo_path / "main.py").read_text()
        results = validator.validate_python_code(test_code)

        # Should validate successfully
        assert results['syntax']['valid'] is True
        assert results['linting'].score > 0.5  # Reasonable quality

        # Should detect framework usage
        imports = results['imports']
        assert 'fastapi' in str(imports).lower()

@pytest.mark.integration
def test_cli_integration():
    """Test CLI commands work together"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        repo_path = Path(temp_dir)
        (repo_path / "app.py").write_text("""
import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}
""")

        # Change to temp directory for CLI commands
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Test mining command
            from cli_module import mine_command
            import argparse

            args = argparse.Namespace(
                repo_path=".",
                emit="test_patterns.json",
                max_files=10
            )

            result = mine_command(args)
            assert result == 0  # Success
            assert Path("test_patterns.json").exists()

            # Test validation command
            from cli_module import validate_command

            args = argparse.Namespace(
                files=["app.py"],
                report="validation.json",
                check=False
            )

            result = validate_command(args)
            assert result == 0  # Success

        finally:
            os.chdir(original_cwd)

@pytest.mark.integration
def test_configuration_integration():
    """Test that configuration system works end-to-end"""

    # Test default configuration
    from code_attractor_system import CodeAttractorSystem

    system = CodeAttractorSystem()

    # Should have default configuration
    assert hasattr(system, 'config')
    assert system.config.model is not None

    # Test compact serialization with configuration
    patterns = list(system.patterns.values())[:3]
    compact_json = system.build_consider_code("test", patterns)

    # Should respect size limits
    full_block = f"<consider>\n{compact_json}\n</consider>"
    size_bytes = len(full_block.encode('utf-8'))

    # Should be under configured limit
    assert size_bytes < system.config.compact_limit

@pytest.mark.integration
def test_github_workflow_simulation():
    """Simulate what would happen in GitHub Actions"""

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Create a PR-like change
        (repo_path / "new_feature.py").write_text("""
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

@app.post("/process")
async def process_data(data: dict):
    try:
        # Process the data
        result = await process_async(data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_async(data):
    # Simulate async processing
    await asyncio.sleep(0.1)
    return {"processed": True}
""")

        # Step 1: Mine patterns (like in CI)
        from repo_mining import RepoMiner
        miner = RepoMiner(str(repo_path))
        patterns = miner.scan_repository()

        # Export patterns (like CI would)
        miner.export_patterns(str(repo_path / "patterns.json"))
        assert (repo_path / "patterns.json").exists()

        # Step 2: Validate code (like in CI)
        from code_validator import CodeValidator
        validator = CodeValidator()

        code = (repo_path / "new_feature.py").read_text()
        results = validator.validate_python_code(code)

        # Generate report (like CI would)
        report = {
            "files_validated": 1,
            "syntax_valid": results['syntax']['valid'],
            "linting_score": results['linting'].score,
            "security_issues": len(results['security']),
            "overall_score": results['linting'].score
        }

        with open(repo_path / "validation_report.json", 'w') as f:
            json.dump(report, f)

        assert (repo_path / "validation_report.json").exists()
        assert report['syntax_valid'] is True
        assert report['overall_score'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])