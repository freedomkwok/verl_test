#!/usr/bin/env python3
"""
Example demonstrating how user_turn_rewards filters agent segments for reward allocation.
"""

def example_user_turn_filtering():
    """Example of how user turn rewards filter agent segments."""
    
    # Example conversation with multiple turns
    segment_positions = [
        {"start": 0, "end": 50, "role": "prompt", "is_agent": False},
        {"start": 51, "end": 80, "role": "assistant", "is_agent": True},   # Agent response 1
        {"start": 81, "end": 100, "role": "user", "is_agent": False},      # User turn 1
        {"start": 101, "end": 150, "role": "assistant", "is_agent": True}, # Agent response 2
        {"start": 151, "end": 170, "role": "user", "is_agent": False},     # User turn 2
        {"start": 171, "end": 220, "role": "assistant", "is_agent": True}, # Agent response 3
    ]
    
    # Example user turn rewards: [positive, negative, positive]
    # This means: User turn 1 is good, User turn 2 is bad, User turn 3 is good
    user_turn_rewards = [0.5, -0.3, 0.8]
    
    print("=== User Turn Reward Filtering Example ===\n")
    print("Segment Positions:")
    for i, seg in enumerate(segment_positions):
        print(f"  {i}: {seg}")
    
    print(f"\nUser Turn Rewards: {user_turn_rewards}")
    print("  - User turn 1: 0.5 (positive - agent response will be rewarded)")
    print("  - User turn 2: -0.3 (negative - agent response will NOT be rewarded)")
    print("  - User turn 3: 0.8 (positive - agent response will be rewarded)")
    
    # Find all agent segments
    all_agent_segments = [seg for seg in segment_positions if seg.get("is_agent", False)]
    print(f"\nAll Agent Segments: {len(all_agent_segments)}")
    for i, seg in enumerate(all_agent_segments):
        print(f"  Agent {i}: tokens {seg['start']}-{seg['end']}")
    
    # Filter agent segments based on user turn rewards
    agent_segments = []
    user_turn_idx = 0
    
    for seg in all_agent_segments:
        if user_turn_idx < len(user_turn_rewards) and user_turn_rewards[user_turn_idx] >= 0:
            agent_segments.append(seg)
            print(f"  ✅ Agent {user_turn_idx} included (user turn reward: {user_turn_rewards[user_turn_idx]})")
        else:
            print(f"  ❌ Agent {user_turn_idx} excluded (user turn reward: {user_turn_rewards[user_turn_idx]})")
        user_turn_idx += 1
    
    print(f"\nFiltered Agent Segments: {len(agent_segments)}")
    for i, seg in enumerate(agent_segments):
        print(f"  Agent {i}: tokens {seg['start']}-{seg['end']}")
    
    print("\n=== Reward Allocation Results ===")
    print("Only the filtered agent segments will receive rewards:")
    print("  - Agent response 1: ✅ Will receive reward (user turn 1 was positive)")
    print("  - Agent response 2: ❌ Will NOT receive reward (user turn 2 was negative)")
    print("  - Agent response 3: ✅ Will receive reward (user turn 3 was positive)")
    
    print("\n✅ This ensures that agent responses to 'bad' user turns don't get rewarded!")

if __name__ == "__main__":
    example_user_turn_filtering()
    print("\n🎉 User turn filtering prevents rewarding responses to negative user interactions!")
