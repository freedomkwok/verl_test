#!/usr/bin/env python3
"""
Simple test runner for GmailInteraction tests.
Run this script to test your GmailInteraction class locally.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.interactions.test_gmail_interaction import TestGmailInteraction


async def run_basic_tests():
    """Run basic tests without pytest."""
    print("🧪 Running GmailInteraction Tests...")
    print("=" * 50)
    
    test_instance = TestGmailInteraction()
    
    try:
        # Test 1: Initialization
        print("1. Testing initialization...")
        test_instance.setup_method()
        test_instance.test_init()
        print("   ✅ Initialization test passed")
        
        # Test 2: Start interaction with mocked server
        print("2. Testing start interaction...")
        with patch('requests.post') as mock_post:
            mock_post.side_effect = [
                Mock(status_code=200, json=lambda: {"env_idx": 1}),
                Mock(status_code=200, json=lambda: {
                    "observation": "Gmail environment ready",
                    "owner_tag": "test_user",
                    "owner_email": "test@example.com",
                    "owner_name": "Test User"
                })
            ]
            instance_id = await test_instance.gmail_interaction.start_interaction(
                ground_truth="Test ground truth"
            )
            print(f"   ✅ Start interaction test passed - Instance ID: {instance_id}")
        
        # Test 3: Tool call handling
        print("3. Testing tool call handling...")
        messages = [
            {"role": "user", "content": "Check my emails"},
            {
                "role": "assistant",
                "content": "I'll check your emails for you.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "gmail_search",
                            "arguments": '{"query": "in:inbox"}'
                        }
                    }
                ]
            }
        ]
        
        should_terminate, response, reward, metadata = await test_instance.gmail_interaction.generate_response(
            instance_id, messages
        )
        print(f"   ✅ Tool call test passed - Response: {response[:50]}...")
        
        # Test 4: Environment stepping
        print("4. Testing environment stepping...")
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: {
                "observation": "Action executed successfully",
                "reward": 0.5,
                "done": False
            })
            
            messages = [
                {"role": "user", "content": "Send an email"},
                {"role": "assistant", "content": "I'll send an email for you."}
            ]
            
            should_terminate, response, reward, metadata = await test_instance.gmail_interaction.generate_response(
                instance_id, messages
            )
            print(f"   ✅ Environment stepping test passed - Reward: {reward}")
        
        # Test 5: Score calculation
        print("5. Testing score calculation...")
        with patch('verl.utils.reward_score.gmail.compute_score', return_value=1.0):
            score = await test_instance.gmail_interaction.calculate_score(instance_id)
            print(f"   ✅ Score calculation test passed - Score: {score}")
        
        # Test 6: Finalization
        print("6. Testing interaction finalization...")
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: {"status": "closed"})
            await test_instance.gmail_interaction.finalize_interaction(instance_id)
            print("   ✅ Finalization test passed")
        
        print("\n🎉 All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_with_pytest():
    """Run tests using pytest if available."""
    try:
        import pytest
        print("🧪 Running tests with pytest...")
        result = pytest.main([
            "tests/interactions/test_gmail_interaction.py",
            "-v",
            "--tb=short"
        ])
        return result == 0
    except ImportError:
        print("⚠️  pytest not available, running basic tests instead...")
        return False


if __name__ == "__main__":
    print("GmailInteraction Test Runner")
    print("=" * 50)
    
    # Try pytest first, fallback to basic tests
    if not run_with_pytest():
        success = asyncio.run(run_basic_tests())
        if not success:
            sys.exit(1)
    
    print("\n✨ Test run completed!")
