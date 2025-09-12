#!/usr/bin/env python3
"""
Example demonstrating the new segment-based reward allocation system.

This shows how the segment_positions data structure enables sophisticated
reward allocation strategies for multi-turn conversations.
"""

import torch

def example_segment_reward_allocation():
    """Example of how segment-based reward allocation works."""
    
    # Example segment_positions from a multi-turn conversation
    segment_positions = [
        {"start": 0, "end": 50, "role": "prompt", "is_agent": False},
        {"start": 51, "end": 80, "role": "assistant", "is_agent": True},  # First agent response
        {"start": 81, "end": 100, "role": "user", "is_agent": False},
        {"start": 101, "end": 150, "role": "assistant", "is_agent": True},  # Second agent response
        {"start": 151, "end": 170, "role": "user", "is_agent": False},
        {"start": 171, "end": 220, "role": "assistant", "is_agent": True},  # Third agent response
    ]
    
    # Create a reward tensor (batch_size=1, seq_len=250)
    reward_tensor = torch.zeros(1, 250)
    valid_response_length = 250
    reward_to_distribute = 1.0
    
    print("=== Segment-Based Reward Allocation Example ===\n")
    print("Segment Positions:")
    for i, seg in enumerate(segment_positions):
        print(f"  {i}: {seg}")
    
    # Find agent segments
    agent_segments = [seg for seg in segment_positions if seg.get("is_agent", False)]
    print(f"\nAgent Segments: {len(agent_segments)}")
    for i, seg in enumerate(agent_segments):
        print(f"  Agent {i}: tokens {seg['start']}-{seg['end']} (length: {seg['end'] - seg['start'] + 1})")
    
    print("\n" + "="*60)
    
    # Test different reward allocation strategies
    strategies = ["last_token", "uniform_positive", "discounted", "uniform_discounted"]
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Strategy ---")
        
        # Reset reward tensor
        reward_tensor.zero_()
        
        # Apply the strategy
        if strategy == "last_token":
            # Assign reward only to the last token of the last agent segment
            last_segment = agent_segments[-1]
            last_token_pos = min(last_segment["end"], valid_response_length - 1)
            reward_tensor[0, last_token_pos] = reward_to_distribute
            
        elif strategy == "uniform_positive":
            # Distribute positive rewards evenly across all agent tokens
            total_agent_tokens = sum(
                min(seg["end"], valid_response_length - 1) - seg["start"] + 1 
                for seg in agent_segments
                if seg["start"] < valid_response_length
            )
            if total_agent_tokens > 0:
                reward_per_token = reward_to_distribute / total_agent_tokens
                for seg in agent_segments:
                    start = seg["start"]
                    end = min(seg["end"], valid_response_length - 1)
                    if start < valid_response_length:
                        reward_tensor[0, start:end+1] = reward_per_token
                        
        elif strategy == "discounted":
            # Distribute reward starting from the last agent segment, discounted backward
            gamma = 0.9
            current_reward = reward_to_distribute
            
            # Iterate segments backward (from last to first)
            for seg in reversed(agent_segments):
                start = seg["start"]
                end = min(seg["end"], valid_response_length - 1)
                
                if start < valid_response_length:
                    segment_len = end - start + 1
                    reward_for_segment = current_reward / segment_len
                    reward_tensor[0, start:end+1] = reward_for_segment
                    
                    # Apply discount for the next (earlier) segment
                    current_reward *= (gamma ** segment_len)
                    
        elif strategy == "uniform_discounted":
            # Combine uniform positive with temporal discounting
            total_agent_tokens = sum(
                min(seg["end"], valid_response_length - 1) - seg["start"] + 1 
                for seg in agent_segments
                if seg["start"] < valid_response_length
            )
            if total_agent_tokens > 0:
                base_reward_per_token = reward_to_distribute / total_agent_tokens
                
                # Apply temporal discounting
                gamma = 0.9
                for i, seg in enumerate(agent_segments):
                    start = seg["start"]
                    end = min(seg["end"], valid_response_length - 1)
                    
                    if start < valid_response_length:
                        # Discount factor based on position (later segments get higher rewards)
                        discount_factor = gamma ** (len(agent_segments) - 1 - i)
                        discounted_reward = base_reward_per_token * discount_factor
                        reward_tensor[0, start:end+1] = discounted_reward
        
        # Show results
        print(f"Total reward distributed: {reward_tensor.sum().item():.4f}")
        print("Reward per agent segment:")
        for i, seg in enumerate(agent_segments):
            start = seg["start"]
            end = min(seg["end"], valid_response_length - 1)
            segment_reward = reward_tensor[0, start:end+1].sum().item()
            print(f"  Agent {i}: {segment_reward:.4f} (tokens {start}-{end})")
        
        # Show reward distribution pattern
        print("Reward distribution pattern (first 10 non-zero rewards):")
        nonzero_indices = torch.nonzero(reward_tensor[0]).flatten()[:10]
        for idx in nonzero_indices:
            print(f"  Token {idx.item()}: {reward_tensor[0, idx].item():.4f}")

if __name__ == "__main__":
    example_segment_reward_allocation()
