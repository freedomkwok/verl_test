#!/usr/bin/env python3
"""
Test script to verify that AsyncRolloutRequest initialization works correctly
with the new segment_positions field.
"""

import torch
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message, TokenizationSanityCheckModeEnum
from transformers import AutoTokenizer

def test_async_rollout_request_initialization():
    """Test that AsyncRolloutRequest initializes correctly with segment_positions."""
    
    # Create a mock processing class (tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Test data
    messages = [Message(role="user", content="Hello, how are you?")]
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Mock token IDs
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
    
    print("Testing AsyncRolloutRequest initialization...")
    
    try:
        # Create AsyncRolloutRequest manually (like in sglang_rollout.py)
        req = AsyncRolloutRequest(
            request_id="test-123",
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=messages,
            tool_schemas=None,
            tools_kwargs={},
            input_ids=input_ids,
            prompt_ids=input_ids,
            response_ids=None,
            attention_mask=attention_mask,
            prompt_attention_mask=attention_mask,
            response_attention_mask=None,
            position_ids=position_ids,
            prompt_position_ids=position_ids,
            response_position_ids=None,
            loss_mask=None,
            prompt_loss_mask=None,
            response_loss_mask=None,
            reward_scores={},
            max_prompt_len=100,
            max_response_len=50,
            max_model_len=150,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.DISABLE,
            processing_class=tokenizer,
        )
        
        print("✅ AsyncRolloutRequest created successfully!")
        print(f"   Request ID: {req.request_id}")
        print(f"   Segment positions: {req.segment_positions}")
        print(f"   Number of segments: {len(req.segment_positions)}")
        
        # Check if prompt segment was initialized
        if req.segment_positions:
            prompt_segment = req.segment_positions[0]
            print(f"   Prompt segment: {prompt_segment}")
            assert prompt_segment["role"] == "prompt"
            assert prompt_segment["is_agent"] == False
            assert prompt_segment["start"] == 0
            print("✅ Prompt segment initialized correctly!")
        else:
            print("❌ No segments found!")
            return False
            
        # Test adding an assistant message
        print("\nTesting add_assistant_message...")
        req.add_assistant_message(tokenizer, "I'm doing well, thank you!")
        
        print(f"   Updated segment positions: {req.segment_positions}")
        print(f"   Number of segments: {len(req.segment_positions)}")
        
        # Check if assistant segment was added
        if len(req.segment_positions) >= 2:
            assistant_segment = req.segment_positions[1]
            print(f"   Assistant segment: {assistant_segment}")
            assert assistant_segment["role"] == "assistant"
            assert assistant_segment["is_agent"] == True
            print("✅ Assistant segment added correctly!")
        else:
            print("❌ Assistant segment not found!")
            return False
            
        print("\n✅ All tests passed! Segment tracking is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_async_rollout_request_initialization()
    if success:
        print("\n🎉 Segment tracking implementation is working correctly!")
    else:
        print("\n💥 There are issues with the segment tracking implementation.")
