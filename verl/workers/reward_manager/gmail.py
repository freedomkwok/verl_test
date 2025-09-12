# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("gmail")
class GmailRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 reward_allocation="discounted", gamma=0.9) -> None:
        """
        Initialize the GmailRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            reward_allocation: The reward allocation strategy. Options: "last_token", "uniform_positive", 
                "discounted", "uniform_discounted". Defaults to "discounted".
            gamma: The discount factor for temporal discounting. Defaults to 0.9.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_allocation = reward_allocation
        self.gamma = gamma  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            user_turn_rewards = data_item.non_tensor_batch["reward_scores"]["user_turn_rewards"]
            total_turns = len(data_item.non_tensor_batch["messages"]["messages"])
            turn_reward = 1 * 0.5 if total_turns < extra_info["conversation_count"] else 0
            segment_positions = data_item.non_tensor_batch["segment_positions"]

            has_finish = user_turn_rewards[-1] >= 0
            original_has_finish = extra_info.get("original_success", False)

            teacher_ground_truth = extra_info.get("teacher_ground_truth", "")
            #teacher_ground_truth match valid_response_ids["last"]

            success_reward = 0.0
            if original_has_finish and has_finish:
                success_reward += 0.5
            elif original_has_finish and not has_finish:
                success_reward -= 0.6
            elif not original_has_finish and has_finish:
                success_reward += 1.0

            score = {
                "score": success_reward + turn_reward,
                "acc": 0,
                "pred": 0,
            }

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # Apply segment-based reward allocation
            self._apply_segment_reward_allocation(
                reward_tensor, i, reward, segment_positions, valid_response_length, user_turn_rewards
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _apply_segment_reward_allocation(
        self, 
        reward_tensor: torch.Tensor, 
        batch_idx: int, 
        reward_to_distribute: float,
        segment_positions: list[dict[str, Any]] | np.ndarray, 
        valid_response_length: int,
        user_turn_rewards: list[float]
    ) -> None:
        """
        Apply segment-based reward allocation using the new segment_positions data structure.
        
        Args:
            reward_tensor: The reward tensor to update
            batch_idx: Index of the current batch item
            reward_to_distribute: The total reward to distribute
            segment_positions: List of segment dictionaries with start, end, role, is_agent
            valid_response_length: Length of the valid response tokens
            user_turn_rewards: List of rewards for each user turn (negative values indicate bad turns)
        """
        # Convert to list if it's a numpy array
        if hasattr(segment_positions, 'tolist'):
            segment_positions = segment_positions.tolist()
        
        if len(segment_positions) == 0:
            # Fallback to last token allocation if no segments
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
            return
        
        # Get reward allocation strategy from instance variables
        reward_allocation = self.reward_allocation
        
        # Find agent response segments
        all_agent_segments = [seg for seg in segment_positions if seg.get("is_agent", False)]
        
        if len(all_agent_segments) == 0:
            # No agent segments found, fallback to last token
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
            return
        
        # Filter out agent segments that correspond to negative user turn rewards
        # The logic: each agent response corresponds to a user turn, and we only reward
        # agent responses if the corresponding user turn has a non-negative reward
        agent_segments = []
        user_turn_idx = 0  # Track which user turn we're on
        
        for seg in all_agent_segments:
            # Check if this agent segment should be rewarded based on user turn rewards
            if user_turn_idx < len(user_turn_rewards) and user_turn_rewards[user_turn_idx] >= 0:
                agent_segments.append(seg)
            user_turn_idx += 1
        
        # Debug logging (can be removed in production)
        if hasattr(self, 'num_examine') and self.num_examine > 0:
            print(f"[Segment Reward Debug] Total agent segments: {len(all_agent_segments)}")
            print(f"[Segment Reward Debug] User turn rewards: {user_turn_rewards}")
            print(f"[Segment Reward Debug] Filtered agent segments: {len(agent_segments)}")
            for i, seg in enumerate(agent_segments):
                print(f"[Segment Reward Debug] Agent segment {i}: tokens {seg['start']}-{seg['end']}")
        
        # If no agent segments remain after filtering, fallback to last token
        if len(agent_segments) == 0:
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
            return

        if reward_allocation == "last_token":
            # Assign reward only to the last token of the last agent segment
            last_segment = agent_segments[-1]
            last_token_pos = min(last_segment["end"], valid_response_length - 1)
            reward_tensor[batch_idx, last_token_pos] = reward_to_distribute
            
        elif reward_allocation == "uniform_positive":
            # Distribute positive rewards evenly across all agent tokens
            if reward_to_distribute > 0:
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
                            reward_tensor[batch_idx, start:end+1] = reward_per_token
            else:
                # Negative rewards go to last token
                last_segment = agent_segments[-1]
                last_token_pos = min(last_segment["end"], valid_response_length - 1)
                reward_tensor[batch_idx, last_token_pos] = reward_to_distribute
                
        elif reward_allocation == "discounted":
            # Distribute reward starting from the last agent segment, discounted backward
            gamma = self.gamma  # Use instance variable
            current_reward = reward_to_distribute
            
            # Iterate segments backward (from last to first)
            for seg in reversed(agent_segments):
                start = seg["start"]
                end = min(seg["end"], valid_response_length - 1)
                
                if start < valid_response_length:
                    segment_len = end - start + 1
                    reward_for_segment = current_reward / segment_len
                    reward_tensor[batch_idx, start:end+1] = reward_for_segment
                    
                    # Apply discount for the next (earlier) segment
                    current_reward *= (gamma ** segment_len)
                    
        elif reward_allocation == "uniform_discounted":
            # Combine uniform positive with temporal discounting
            if reward_to_distribute > 0:
                # Apply uniform positive to all agent segments
                total_agent_tokens = sum(
                    min(seg["end"], valid_response_length - 1) - seg["start"] + 1 
                    for seg in agent_segments
                    if seg["start"] < valid_response_length
                )
                if total_agent_tokens > 0:
                    base_reward_per_token = reward_to_distribute / total_agent_tokens
                    
                    # Apply temporal discounting
                    gamma = self.gamma  # Use instance variable
                    for i, seg in enumerate(agent_segments):
                        start = seg["start"]
                        end = min(seg["end"], valid_response_length - 1)
                        
                        if start < valid_response_length:
                            # Discount factor based on position (later segments get higher rewards)
                            discount_factor = gamma ** (len(agent_segments) - 1 - i)
                            discounted_reward = base_reward_per_token * discount_factor
                            reward_tensor[batch_idx, start:end+1] = discounted_reward
            else:
                # Negative rewards go to last token
                last_segment = agent_segments[-1]
                last_token_pos = min(last_segment["end"], valid_response_length - 1)
                reward_tensor[batch_idx, last_token_pos] = reward_to_distribute
        else:
            # Unknown strategy, fallback to last token
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
