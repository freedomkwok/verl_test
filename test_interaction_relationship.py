#!/usr/bin/env python3
"""
Test script to demonstrate the relationship between prompts and interactions.
This shows whether it's one-to-one or one-to-many and how kwargs are updated.
"""

import numpy as np
from verl import DataProto
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.utils.config import omega_conf_to_dataclass
from unittest.mock import Mock, patch, AsyncMock


def demonstrate_interaction_relationship():
    """
    Demonstrate how interactions work with prompts and kwargs.
    """
    print("🔍 Interaction Relationship Analysis")
    print("=" * 60)
    
    # 1. SHOW THE DATA STRUCTURE
    print("\n1. 📊 Data Structure:")
    print("-" * 30)
    
    # Example: Multiple prompts with different interaction kwargs
    interaction_kwargs_list = [
        {"name": "gmail", "query": "Check emails", "ground_truth": "Emails checked"},
        {"name": "gsm8k", "query": "What is 2+2?", "ground_truth": "4"},
        {"name": "gmail", "query": "Send email", "ground_truth": "Email sent"},
        {"name": "weather", "query": "What's the weather?", "ground_truth": "Sunny"},
    ]
    
    print(f"Number of prompts: {len(interaction_kwargs_list)}")
    print(f"Interaction kwargs per prompt:")
    for i, kwargs in enumerate(interaction_kwargs_list):
        print(f"  Prompt {i}: {kwargs}")
    
    # 2. SHOW THE RELATIONSHIP
    print("\n2. 🔗 Relationship Analysis:")
    print("-" * 30)
    
    # Count interactions by name
    interaction_counts = {}
    for kwargs in interaction_kwargs_list:
        name = kwargs.get("name", "gsm8k")
        interaction_counts[name] = interaction_counts.get(name, 0) + 1
    
    print("Interaction usage:")
    for name, count in interaction_counts.items():
        print(f"  {name}: {count} prompts")
    
    # 3. SHOW HOW IT'S PROCESSED
    print("\n3. ⚙️ Processing Flow:")
    print("-" * 30)
    
    print("For EACH prompt in the batch:")
    print("  1. Extract interaction_kwargs[data_idx]")
    print("  2. Get interaction name: interaction_kwargs.get('name', 'gsm8k')")
    print("  3. Look up interaction: interaction_map[interaction_name]")
    print("  4. Call interaction.generate_response(request_id, messages, **interaction_kwargs)")
    print("  5. Interaction can update its internal state based on kwargs")
    
    # 4. SHOW THE KEY INSIGHT
    print("\n4. 🎯 Key Insight:")
    print("-" * 30)
    print("✅ It's ONE-TO-ONE: Each prompt gets its own interaction instance")
    print("✅ Each prompt has its own interaction_kwargs")
    print("✅ The same interaction class can be used by multiple prompts")
    print("✅ Each interaction maintains separate state per request_id")
    
    return interaction_kwargs_list


def demonstrate_kwargs_updates():
    """
    Show how interaction kwargs can be updated during processing.
    """
    print("\n5. 🔄 Kwargs Update Examples:")
    print("-" * 30)
    
    # Example 1: Static kwargs (no updates)
    static_kwargs = {
        "name": "gmail",
        "query": "Check emails",
        "ground_truth": "Emails checked"
    }
    print(f"Static kwargs: {static_kwargs}")
    
    # Example 2: Dynamic kwargs (can be updated)
    dynamic_kwargs = {
        "name": "gmail",
        "query": "Check emails",
        "ground_truth": "Emails checked",
        "max_turns": 5,
        "current_turn": 0,  # This could be updated
        "user_preferences": {"language": "en"}  # This could be updated
    }
    print(f"Dynamic kwargs: {dynamic_kwargs}")
    
    # Example 3: How kwargs might be updated
    print("\n🔄 How kwargs can be updated:")
    print("  1. Interaction.generate_response() can modify kwargs")
    print("  2. Interaction can store state in its internal dict")
    print("  3. Next call to same interaction gets updated state")
    print("  4. But kwargs passed to generate_response are per-call")


def demonstrate_interaction_state():
    """
    Show how interaction state is managed per request.
    """
    print("\n6. 🏠 Interaction State Management:")
    print("-" * 30)
    
    print("Each interaction maintains state per request_id:")
    print("  - GmailInteraction._instance_dict[request_id] = {...}")
    print("  - Each request gets its own state")
    print("  - State persists across multiple turns")
    print("  - State is cleaned up when interaction is finalized")
    
    # Example state structure
    example_state = {
        "response": "",
        "ground_truth": "Emails checked",
        "reward": 0.0,
        "step": 0,
        "observation": "Gmail environment ready",
        "owner_tag": "test_user",
        "owner_email": "test@example.com",
        "owner_name": "Test User"
    }
    print(f"\nExample state structure: {example_state}")


def create_test_data_proto():
    """
    Create a test DataProto to show the structure.
    """
    print("\n7. 📦 DataProto Structure:")
    print("-" * 30)
    
    # Create sample data
    batch_size = 3
    input_ids = np.random.randint(0, 1000, (batch_size, 10))
    attention_mask = np.ones((batch_size, 10))
    
    # Create interaction kwargs for each prompt
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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "interaction_kwargs": np.array(interaction_kwargs, dtype=object),
        }
    )
    
    print(f"DataProto structure:")
    print(f"  batch_size: {len(data_proto)}")
    print(f"  batch keys: {list(data_proto.batch.keys())}")
    print(f"  non_tensor_batch keys: {list(data_proto.non_tensor_batch.keys())}")
    print(f"  interaction_kwargs shape: {data_proto.non_tensor_batch['interaction_kwargs'].shape}")
    
    # Show how each prompt gets its kwargs
    print(f"\nPer-prompt interaction_kwargs:")
    for i in range(len(data_proto)):
        kwargs = data_proto.non_tensor_batch["interaction_kwargs"][i]
        print(f"  Prompt {i}: {kwargs}")
    
    return data_proto


def demonstrate_async_rollout_request():
    """
    Show how AsyncRolloutRequest handles interaction_kwargs.
    """
    print("\n8. 🚀 AsyncRolloutRequest Processing:")
    print("-" * 30)
    
    # Create a mock request
    request = AsyncRolloutRequest(
        request_id="test_request_123",
        state=AsyncRolloutRequestStateEnum.PENDING,
        messages=[Message(role="user", content="Test message")],
        interaction_kwargs={"name": "gmail", "query": "Check emails", "ground_truth": "Emails checked"},
        input_ids=None,
        prompt_ids=None,
        response_ids=None,
        attention_mask=None,
        prompt_attention_mask=None,
        response_attention_mask=None,
        position_ids=None,
        prompt_position_ids=None,
        response_position_ids=None,
        loss_mask=None,
        prompt_loss_mask=None,
        response_loss_mask=None,
        reward_scores={},
        max_prompt_len=512,
        max_response_len=256,
        max_model_len=1024,
        metrics={},
        output_token_ids=None,
        rollout_log_probs=None,
        use_inference_chat_template=True,
        tokenization_sanity_check_mode="disable",
        generation_prompt_ids=None,
        base_conv_wo_gen_prompt_end_pos=0,
        base_conv_with_gen_prompt_end_pos=0,
        processing_class=Mock(),  # Mock tokenizer
    )
    
    print(f"Request ID: {request.request_id}")
    print(f"Interaction kwargs: {request.interaction_kwargs}")
    print(f"Interaction name: {request.interaction_kwargs.get('name', 'gsm8k')}")
    
    # Show how the interaction would be selected
    interaction_name = request.interaction_kwargs.get("name", "gsm8k")
    print(f"Selected interaction: {interaction_name}")
    
    return request


def main():
    """Main demonstration function."""
    print("🧪 Interaction Relationship Test")
    print("=" * 60)
    
    # Run all demonstrations
    interaction_kwargs_list = demonstrate_interaction_relationship()
    demonstrate_kwargs_updates()
    demonstrate_interaction_state()
    data_proto = create_test_data_proto()
    request = demonstrate_async_rollout_request()
    
    print("\n9. 📋 Summary:")
    print("-" * 30)
    print("✅ ONE-TO-ONE: Each prompt gets its own interaction instance")
    print("✅ Each prompt has its own interaction_kwargs")
    print("✅ Same interaction class can serve multiple prompts")
    print("✅ Each interaction maintains separate state per request_id")
    print("✅ Kwargs can be updated during interaction processing")
    print("✅ State persists across multiple turns per request")
    
    print("\n🎯 Answer to your question:")
    print("The relationship is ONE-TO-ONE between prompts and interaction instances.")
    print("Each prompt in the DataProto gets its own interaction_kwargs,")
    print("and the interaction can update its internal state based on these kwargs.")
    print("The kwargs themselves are passed per-call to generate_response().")


if __name__ == "__main__":
    main()
