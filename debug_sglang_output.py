#!/usr/bin/env python3
"""
Debug script to investigate SGLangRollout output issues.
This will help identify why the decoded output is garbled.
"""

import torch
import asyncio
from typing import Optional, List, Dict, Any
from unittest.mock import Mock, patch

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.utils.config import omega_conf_to_dataclass


class SGLangOutputDebugger:
    """Debug helper for SGLangRollout output issues."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.rollout = None
        
    async def setup_rollout(self):
        """Set up the SGLangRollout instance."""
        print("🔧 Setting up SGLangRollout...")
        
        # Create basic rollout config
        rollout_config_dict = {
            "name": "debug_test",
            "mode": "async",
            "load_format": "dummy",
            "enforce_eager": False,
            "free_cache_engine": True,
            "dtype": "float16",
            "tensor_model_parallel_size": 1,
            "nnodes": 1,
            "node_rank": 0,
            "gpu_memory_utilization": 0.5,
            "ignore_eos": False,
            "max_num_batched_tokens": 8192,
            "max_response_length": 512,
            "max_prompt_length": 2048,
            "max_model_len": 4096,
            "use_inference_chat_template": True,
            "tokenization_sanity_check_mode": "disable",
        }
        
        rollout_config = omega_conf_to_dataclass(RolloutConfig, rollout_config_dict)
        model_config = HFModelConfig(path=self.model_path)
        
        self.rollout = SGLangRollout(
            config=rollout_config,
            model_config=model_config,
            device_mesh=None,
        )
        
        print("✅ SGLangRollout setup complete")
        
    def debug_tokenizer(self):
        """Debug tokenizer functionality."""
        print("\n🔍 Debugging Tokenizer...")
        print("=" * 50)
        
        processing_class = self.rollout.processing_class
        
        # Test basic tokenization
        test_text = "Hello, how are you?"
        print(f"Test text: '{test_text}'")
        
        # Encode
        input_ids = processing_class.encode(test_text, return_tensors="pt")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs: {input_ids}")
        
        # Decode
        decoded_text = processing_class.decode(input_ids[0], skip_special_tokens=True)
        print(f"Decoded text: '{decoded_text}'")
        
        # Check if round-trip works
        if decoded_text.strip() == test_text.strip():
            print("✅ Tokenizer round-trip works correctly")
        else:
            print("❌ Tokenizer round-trip failed!")
            print(f"  Original: '{test_text}'")
            print(f"  Decoded:  '{decoded_text}'")
            
        # Test with special tokens
        print("\n🔍 Testing with special tokens...")
        decoded_with_special = processing_class.decode(input_ids[0], skip_special_tokens=False)
        print(f"Decoded with special tokens: '{decoded_with_special}'")
        
        # Check vocabulary
        print(f"\n📊 Tokenizer info:")
        print(f"  Vocab size: {processing_class.vocab_size}")
        print(f"  Model max length: {getattr(processing_class, 'model_max_length', 'Unknown')}")
        print(f"  Pad token: {processing_class.pad_token}")
        print(f"  EOS token: {processing_class.eos_token}")
        print(f"  BOS token: {processing_class.bos_token}")
        
    def debug_chat_template(self):
        """Debug chat template functionality."""
        print("\n🔍 Debugging Chat Template...")
        print("=" * 50)
        
        processing_class = self.rollout.processing_class
        
        # Test chat template
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        try:
            # Test without tools
            template_result = processing_class.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"Chat template result: '{template_result}'")
            
            # Test with tokenization
            template_ids = processing_class.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            )
            print(f"Template IDs shape: {template_ids.shape}")
            print(f"Template IDs: {template_ids}")
            
            # Decode the template
            decoded_template = processing_class.decode(template_ids[0], skip_special_tokens=True)
            print(f"Decoded template: '{decoded_template}'")
            
        except Exception as e:
            print(f"❌ Chat template error: {e}")
            
    def debug_model_output(self):
        """Debug model output generation."""
        print("\n🔍 Debugging Model Output...")
        print("=" * 50)
        
        processing_class = self.rollout.processing_class
        
        # Create a simple prompt
        messages = [
            {"role": "user", "content": "Say hello"}
        ]
        
        try:
            # Get generation prompt IDs
            generation_prompt_ids = processing_class.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            print(f"Generation prompt IDs shape: {generation_prompt_ids.shape}")
            print(f"Generation prompt IDs: {generation_prompt_ids}")
            
            # Decode the prompt
            prompt_text = processing_class.decode(generation_prompt_ids[0], skip_special_tokens=True)
            print(f"Prompt text: '{prompt_text}'")
            
            # Check if the model engine exists
            if hasattr(self.rollout, '_engine') and self.rollout._engine is not None:
                print("✅ Model engine is available")
                
                # Try a simple generation
                print("\n🧪 Testing model generation...")
                
                # Mock the engine generate call to see what it returns
                with patch.object(self.rollout._engine, 'async_generate') as mock_generate:
                    # Create a mock response
                    mock_output = {
                        "output_ids": torch.tensor([[1, 2, 3, 4, 5]]),  # Simple test tokens
                        "finish_reason": "stop"
                    }
                    mock_generate.return_value = mock_output
                    
                    # Test the generation
                    result = await self.rollout._handle_engine_generate(
                        generation_prompt_ids.squeeze(0).tolist(),
                        {"max_tokens": 10, "temperature": 0.7}
                    )
                    
                    print(f"Mock generation result: {result}")
                    
                    # Test decoding
                    if "output_ids" in result:
                        decoded_output = processing_class.decode(
                            result["output_ids"], 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        print(f"Decoded mock output: '{decoded_output}'")
                        
            else:
                print("❌ Model engine is not available")
                
        except Exception as e:
            print(f"❌ Model output debug error: {e}")
            import traceback
            traceback.print_exc()
            
    def debug_token_ids(self, token_ids: torch.Tensor):
        """Debug specific token IDs that are causing issues."""
        print("\n🔍 Debugging Problematic Token IDs...")
        print("=" * 50)
        
        processing_class = self.rollout.processing_class
        
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Token IDs: {token_ids}")
        
        # Check for invalid token IDs
        vocab_size = processing_class.vocab_size
        invalid_tokens = token_ids[token_ids >= vocab_size]
        if len(invalid_tokens) > 0:
            print(f"❌ Found {len(invalid_tokens)} invalid token IDs (>= {vocab_size})")
            print(f"Invalid tokens: {invalid_tokens}")
        else:
            print("✅ All token IDs are within vocabulary range")
            
        # Try different decoding options
        print("\n🔍 Testing different decoding options...")
        
        # Option 1: Skip special tokens
        decoded_1 = processing_class.decode(token_ids, skip_special_tokens=True)
        print(f"Decoded (skip_special_tokens=True): '{decoded_1[:100]}...'")
        
        # Option 2: Keep special tokens
        decoded_2 = processing_class.decode(token_ids, skip_special_tokens=False)
        print(f"Decoded (skip_special_tokens=False): '{decoded_2[:100]}...'")
        
        # Option 3: With clean up
        decoded_3 = processing_class.decode(
            token_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        print(f"Decoded (with clean_up): '{decoded_3[:100]}...'")
        
        # Option 4: Individual token decoding
        print("\n🔍 Individual token analysis...")
        for i, token_id in enumerate(token_ids[:10]):  # First 10 tokens
            try:
                token_text = processing_class.decode([token_id], skip_special_tokens=True)
                print(f"Token {i}: ID={token_id}, Text='{token_text}'")
            except Exception as e:
                print(f"Token {i}: ID={token_id}, Error: {e}")


async def main():
    """Main debug function."""
    print("🐛 SGLangRollout Output Debugger")
    print("=" * 50)
    
    # You can change this to your model path
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
    
    debugger = SGLangOutputDebugger(model_path)
    
    try:
        # Setup
        await debugger.setup_rollout()
        
        # Run debug tests
        debugger.debug_tokenizer()
        debugger.debug_chat_template()
        await debugger.debug_model_output()
        
        # If you have specific problematic token IDs, test them here
        # debugger.debug_token_ids(your_problematic_token_ids)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
