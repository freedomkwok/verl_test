#!/usr/bin/env python3
"""
Simple test script to run the SGLangRollout debugger.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from debug_sglang_output import SGLangOutputDebugger


async def main():
    """Run the debugger with your model."""
    print("🐛 SGLangRollout Debug Test")
    print("=" * 50)
    
    # Change this to your actual model path
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
    
    print(f"Testing with model: {model_path}")
    print("(Change the model_path variable to test with your actual model)")
    
    debugger = SGLangOutputDebugger(model_path)
    
    try:
        await debugger.setup_rollout()
        debugger.debug_tokenizer()
        debugger.debug_chat_template()
        await debugger.debug_model_output()
        
        print("\n✅ Debug completed successfully!")
        print("\nIf you see garbled output, try running:")
        print("python fix_sglang_output.py")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
