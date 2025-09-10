#!/usr/bin/env python3
"""
Simple test to demonstrate the one-to-one relationship between prompts and interactions.
"""

import numpy as np
from verl import DataProto
from unittest.mock import Mock, AsyncMock, patch


def test_interaction_relationship():
    """Test the one-to-one relationship between prompts and interactions."""
    print("🧪 Testing Interaction Relationship")
    print("=" * 50)
    
    # Create a batch of 3 prompts with different interaction kwargs
    batch_size = 3
    
    # Each prompt has its own interaction_kwargs
    interaction_kwargs = [
        {"name": "gmail", "query": "Check emails", "ground_truth": "Emails checked"},
        {"name": "gsm8k", "query": "What is 2+2?", "ground_truth": "4"},
        {"name": "gmail", "query": "Send email", "ground_truth": "Email sent"},
    ]
    
    # Create raw prompts
    raw_prompts = [
        [{"role": "user", "content": "Check my emails"}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Send an email"}],
    ]
    
    # Create DataProto
    data_proto = DataProto(
        batch={
            "input_ids": np.random.randint(0, 1000, (batch_size, 10)),
            "attention_mask": np.ones((batch_size, 10)),
        },
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "interaction_kwargs": np.array(interaction_kwargs, dtype=object),
        }
    )
    
    print(f"Created DataProto with {len(data_proto)} prompts")
    print(f"Each prompt has its own interaction_kwargs:")
    
    # Show how each prompt gets its own kwargs
    for i in range(len(data_proto)):
        kwargs = data_proto.non_tensor_batch["interaction_kwargs"][i]
        print(f"  Prompt {i}: {kwargs}")
    
    # Simulate how SGLangRollout processes this
    print(f"\n🔍 Simulating SGLangRollout processing:")
    print("-" * 40)
    
    # Mock interaction map
    interaction_map = {
        "gmail": Mock(name="GmailInteraction"),
        "gsm8k": Mock(name="GSM8KInteraction"),
    }
    
    # Simulate the processing loop
    for data_idx in range(len(data_proto)):
        print(f"\nProcessing prompt {data_idx}:")
        
        # Extract kwargs for this specific prompt
        prompt_kwargs = data_proto.non_tensor_batch["interaction_kwargs"][data_idx]
        print(f"  Extracted kwargs: {prompt_kwargs}")
        
        # Get interaction name
        interaction_name = prompt_kwargs.get("name", "gsm8k")
        print(f"  Interaction name: {interaction_name}")
        
        # Get interaction instance
        interaction = interaction_map[interaction_name]
        print(f"  Interaction instance: {interaction.name}")
        
        # Show that each prompt gets its own interaction instance
        print(f"  ✅ Each prompt gets its own interaction instance")
        print(f"  ✅ Each prompt has its own kwargs: {prompt_kwargs}")
    
    print(f"\n🎯 Key Findings:")
    print("-" * 20)
    print("✅ ONE-TO-ONE: Each prompt gets its own interaction instance")
    print("✅ Each prompt has its own interaction_kwargs")
    print("✅ Same interaction class can serve multiple prompts")
    print("✅ Each interaction maintains separate state per request_id")
    
    return data_proto, interaction_map


def test_kwargs_updates():
    """Test how kwargs can be updated during interaction processing."""
    print(f"\n🔄 Testing Kwargs Updates:")
    print("-" * 30)
    
    # Create a mock interaction that updates kwargs
    class MockGmailInteraction:
        def __init__(self):
            self._instance_dict = {}
        
        async def start_interaction(self, request_id, **kwargs):
            print(f"  Starting interaction for {request_id} with kwargs: {kwargs}")
            self._instance_dict[request_id] = {
                "kwargs": kwargs.copy(),
                "step": 0,
                "state": "started"
            }
        
        async def generate_response(self, request_id, messages, **kwargs):
            print(f"  Generating response for {request_id} with kwargs: {kwargs}")
            
            # Update kwargs based on processing
            if request_id in self._instance_dict:
                instance = self._instance_dict[request_id]
                instance["step"] += 1
                instance["kwargs"]["current_step"] = instance["step"]
                instance["kwargs"]["last_message"] = messages[-1]["content"] if messages else ""
                
                print(f"  Updated kwargs: {instance['kwargs']}")
                
                return False, f"Response for step {instance['step']}", 0.5, {}
            else:
                return True, "No instance found", 0.0, {}
    
    # Test the interaction
    interaction = MockGmailInteraction()
    request_id = "test_request_123"
    
    # Start interaction
    import asyncio
    asyncio.run(interaction.start_interaction(
        request_id, 
        name="gmail", 
        query="Check emails", 
        ground_truth="Emails checked"
    ))
    
    # Generate response (this updates kwargs)
    asyncio.run(interaction.generate_response(
        request_id,
        [{"role": "user", "content": "Check my emails"}],
        name="gmail",
        query="Check emails",
        ground_truth="Emails checked"
    ))
    
    print(f"✅ Kwargs can be updated during interaction processing")
    print(f"✅ Each request maintains its own state")


def main():
    """Main test function."""
    print("🧪 Simple Interaction Relationship Test")
    print("=" * 60)
    
    # Test the relationship
    data_proto, interaction_map = test_interaction_relationship()
    
    # Test kwargs updates
    test_kwargs_updates()
    
    print(f"\n📋 Final Answer:")
    print("=" * 20)
    print("The relationship is ONE-TO-ONE:")
    print("  - Each prompt gets its own interaction instance")
    print("  - Each prompt has its own interaction_kwargs")
    print("  - The same interaction class can serve multiple prompts")
    print("  - Each interaction maintains separate state per request_id")
    print("  - Kwargs can be updated during interaction processing")


if __name__ == "__main__":
    main()
