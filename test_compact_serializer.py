#!/usr/bin/env python3
"""
Test the compact serializer to ensure it stays under 350B
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tag_injection_enhanced import EnhancedTagInjector

def test_compact_serializer():
    """Test that compact serializer stays under 350B"""

    # Set dummy API key for testing
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Create test injector
    injector = EnhancedTagInjector()

    # Test with various scenarios
    test_cases = [
        {
            "theme": "coffee_mobile_service",
            "cards": [
                {
                    "name": "coffee_bicycle",
                    "summary": "Mobile coffee service combining cafe culture with sustainable transport for urban environments",
                    "neighbors": ["coffee", "bicycle", "mobile", "service", "sustainability", "urban", "transport"],
                    "confidence": 0.92
                },
                {
                    "name": "permit_system",
                    "summary": "Regulatory framework for mobile food service operations requiring licenses and health permits",
                    "neighbors": ["permit", "license", "health", "regulation", "mobile", "food"],
                    "confidence": 0.85
                },
                {
                    "name": "route_optimization",
                    "summary": "Dynamic path planning for mobile services based on demand patterns and traffic",
                    "neighbors": ["route", "optimization", "demand", "traffic", "planning", "dynamic"],
                    "confidence": 0.78
                }
            ]
        },
        {
            "theme": "api_development",
            "cards": [
                {
                    "name": "fastapi_pattern",
                    "summary": "RESTful API pattern with FastAPI and async operations for high-performance web services",
                    "neighbors": ["fastapi", "async", "pydantic", "route", "dependency", "rest", "api"],
                    "confidence": 0.95
                },
                {
                    "name": "middleware_auth",
                    "summary": "Authentication middleware pattern for securing API endpoints with JWT tokens",
                    "neighbors": ["middleware", "auth", "jwt", "token", "security", "endpoint"],
                    "confidence": 0.88
                }
            ]
        }
    ]

    print("🧪 Testing Compact Serializer")
    print("=" * 50)

    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['theme']}")

        # Generate compact block
        compact_json = injector.compact_consider(
            test_case['theme'],
            test_case['cards']
        )

        # Create full consider block
        full_block = f"<consider>\n{compact_json}\n</consider>"

        # Measure size
        size_bytes = len(full_block.encode('utf-8'))

        print(f"   Size: {size_bytes} bytes")
        print(f"   Block: {full_block[:100]}...")

        # Check if under 350B
        if size_bytes <= 350:
            print(f"   ✅ PASS: Under 350B limit")
        else:
            print(f"   ❌ FAIL: Exceeds 350B limit")
            all_passed = False

        # Show actual compact JSON
        print(f"   Compact: {compact_json}")

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Compact serializer working correctly.")
    else:
        print("⚠️  Some tests failed. Review serializer logic.")

    return all_passed

if __name__ == "__main__":
    test_compact_serializer()