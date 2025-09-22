#!/usr/bin/env python3
"""
Two-Pass Decode Helper for Knowledge Attractors
First pass extracts key concepts, second pass uses them for full generation
"""

import requests
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class TwoPassConfig:
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    first_pass_tokens: int = 50
    extract_count: int = 3

class TwoPassDecoder:
    """Two-pass decoder for better concept integration"""

    def __init__(self, config: TwoPassConfig = None):
        self.config = config or TwoPassConfig()

    def generate_with_ollama(self, prompt: str, system: str = None, max_tokens: int = None) -> str:
        """Generate response using Ollama"""
        url = f"{self.config.base_url}/api/generate"

        options = {"temperature": 0.7}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('response', 'No response')
        except Exception as e:
            return f"Error: {e}"

    def extract_concepts(self, text: str, count: int = 3) -> List[str]:
        """Extract key nouns/verbs from text"""
        # Simple extraction: find capitalized words and common action words
        words = re.findall(r'\b[A-Z][a-z]+|\b(?:build|create|design|implement|optimize|serve|provide|enable|support|integrate)\b', text)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_words.append(word)

        return unique_words[:count]

    def two_pass_generate(self, prompt: str, consider_block: str) -> Tuple[str, Dict]:
        """
        Two-pass generation:
        1. Short pass to extract concepts
        2. Full pass with extracted concepts as hidden notes
        """

        # First pass: short generation with consider block
        first_system = (
            f"{consider_block}\n\n"
            "Generate a brief response focusing on key concepts."
        )

        first_response = self.generate_with_ollama(
            prompt,
            first_system,
            max_tokens=self.config.first_pass_tokens
        )

        # Extract key concepts
        extracted_concepts = self.extract_concepts(first_response, self.config.extract_count)

        if not extracted_concepts:
            # Fall back to regular generation
            full_system = (
                f"{consider_block}\n\n"
                "Use the consider block to guide your answer. "
                "Integrate at least one concrete mechanism inspired by it."
            )
            full_response = self.generate_with_ollama(prompt, full_system)
            return full_response, {
                "first_pass": first_response,
                "extracted_concepts": [],
                "method": "fallback"
            }

        # Second pass: full generation with extracted concepts as hidden notes
        notes_line = f"notes: {', '.join(extracted_concepts)}"

        second_system = (
            f"{consider_block}\n\n"
            "Use the consider block to guide your answer. "
            "Integrate at least one concrete mechanism inspired by it. "
            "Focus on practical implementation details."
        )

        # Create assistant notes prefix (hidden from user)
        full_prompt = f"{notes_line}\n\nUser: {prompt}"

        full_response = self.generate_with_ollama(full_prompt, second_system)

        return full_response, {
            "first_pass": first_response,
            "extracted_concepts": extracted_concepts,
            "notes_line": notes_line,
            "method": "two_pass"
        }

def demo_two_pass():
    """Demonstrate two-pass decoding"""

    print("🔄 Two-Pass Decode Demo")
    print("=" * 50)

    decoder = TwoPassDecoder()

    # Example consider block (compact format)
    consider_block = '''<consider>
{"t":"mobile_coffee","a":[{"c":"coffee_bike","h":"Mobile coffee service combining cafe","n":["coffee","bicycle"]},{"c":"permit_sys","h":"Regulatory framework for mobile foo","n":["permit","license"]}]}
</consider>'''

    test_prompt = "How can entrepreneurs start a mobile coffee business in urban areas?"

    print(f"Prompt: {test_prompt}")
    print(f"Consider block size: {len(consider_block)} bytes")

    # Run two-pass generation
    result, metadata = decoder.two_pass_generate(test_prompt, consider_block)

    print(f"\n📝 First pass ({decoder.config.first_pass_tokens} tokens):")
    print(f"   {metadata['first_pass']}")

    print(f"\n🔍 Extracted concepts:")
    print(f"   {metadata['extracted_concepts']}")

    if metadata['method'] == 'two_pass':
        print(f"\n📋 Hidden notes line:")
        print(f"   {metadata['notes_line']}")

    print(f"\n📝 Final response:")
    print(f"   {result}")

    print(f"\n✅ Method: {metadata['method']}")

if __name__ == "__main__":
    demo_two_pass()