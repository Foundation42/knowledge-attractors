.PHONY: eval demo-finance demo-bio demo-code test clean install help

# Default target
help:
	@echo "Knowledge Attractors - Weaponized Resonance Loop"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make eval          - Run hard evaluation scoreboard"
	@echo "  make demo-finance  - Run finance domain demo"
	@echo "  make demo-bio      - Run biology domain demo"
	@echo "  make demo-code     - Run code pattern demo"
	@echo "  make test          - Run all tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make install       - Install dependencies"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make eval"

# Install dependencies
install:
	pip install numpy torch scikit-learn matplotlib seaborn tqdm

# Run evaluation scoreboard
eval:
	@echo "🎯 Running Hard Evaluation Scoreboard..."
	@python eval_scoreboard.py
	@echo ""
	@echo "📊 Results saved to:"
	@echo "   - eval_scoreboard.json (metrics)"
	@echo "   - eval_radar.png (visualization)"

# Domain demos
demo-finance:
	@echo "💰 Running Finance Domain Demo..."
	@python domain_demos.py --domain finance

demo-bio:
	@echo "🧬 Running Biology Domain Demo..."
	@python domain_demos.py --domain bio

demo-code:
	@echo "💻 Running Code Pattern Demo..."
	@python domain_demos.py --domain code

# Run all tests
test:
	@echo "🧪 Running Test Suite..."
	@python test_resonance_hardening.py
	@echo ""
	@echo "✅ All tests complete"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f resonance_visualization*.png
	@rm -f resonance_log*.json
	@rm -f resonance_persistence.json
	@rm -f eval_scoreboard.json eval_radar.png
	@rm -f brave_exploration_diary.json
	@echo "✅ Clean complete"

# Quick demo
demo:
	@echo "🚀 Running Complete Resonance Demo..."
	@python demo_resonance_complete.py
