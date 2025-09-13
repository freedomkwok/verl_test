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
                 reward_allocation="last_token", gamma=0.9) -> None:
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
            turn_reward = 0.5 if total_turns <= extra_info["conversation_count"] else 0
            segment_positions = data_item.non_tensor_batch["segment_positions"]

            has_finish = user_turn_rewards[-1] >= 0 if len(user_turn_rewards) > 0 else False
            original_has_finish = extra_info.get("original_success", False)

            teacher_ground_truth = extra_info.get("teacher_ground_truth", "")
            #teacher_ground_truth match valid_response_ids["last"]

            success_reward = 0.0
            if original_has_finish and has_finish:
                success_reward += 0.8
            elif original_has_finish and not has_finish:
                success_reward -= 0.6
            elif not original_has_finish and has_finish:
                success_reward += 2.0

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
                reward_tensor, i, reward, segment_positions, valid_response_length, user_turn_rewards, prompt_length
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
        valid_response_length: int | torch.Tensor,
        user_turn_rewards: list[float],
        prompt_length: int
    ) -> None:
        """
        Apply segment-based reward allocation using the new segment_positions data structure.
        
        Args:
            reward_tensor: The reward tensor to update
            batch_idx: Index of the current batch item
            reward_to_distribute: The total reward to distribute
            segment_positions: List of segment dictionaries with start, end, role, is_agent (absolute positions)
            valid_response_length: Length of the valid response tokens (int or torch.Tensor)
            user_turn_rewards: List of rewards for each user turn (negative values indicate bad turns)
            prompt_length: Length of the prompt (used to convert absolute to relative positions)
        """
        # Convert to list if it's a numpy array
        if hasattr(segment_positions, 'tolist'):
            segment_positions = segment_positions.tolist()
        
        # Convert tensor to scalar if needed
        if hasattr(valid_response_length, 'item'):
            valid_response_length = valid_response_length.item()
        
        if len(segment_positions) == 0:
            # Fallback to last token allocation if no segments
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
            return
        
        # Convert absolute positions to relative positions (relative to response start)
        # and filter out segments that are not in the response part
        response_segments = []
        for seg in segment_positions:
            # Convert absolute positions to relative positions
            relative_start = seg["start"] - prompt_length
            relative_end = seg["end"] - prompt_length
            
            # Only include segments that are in the response part (after prompt)
            if relative_start >= 0 and relative_end < valid_response_length:
                response_segments.append({
                    "start": relative_start,
                    "end": relative_end,
                    "role": seg["role"],
                    "is_agent": seg["is_agent"]
                })
        
        
        # Find agent response segments
        all_agent_segments = [seg for seg in response_segments if seg.get("is_agent", False)]
        
        # Debug logging (can be removed in production)
        if hasattr(self, 'num_examine') and self.num_examine > 0:
            print(f"[Segment Reward Debug] Prompt length: {prompt_length}")
            print(f"[Segment Reward Debug] Valid response length: {valid_response_length}")
            print(f"[Segment Reward Debug] Total response segments: {len(response_segments)}")
            print(f"[Segment Reward Debug] Total agent segments: {len(all_agent_segments)}")
            print(f"[Segment Reward Debug] User turn rewards: {user_turn_rewards}")
            for i, seg in enumerate(all_agent_segments):
                reward = user_turn_rewards[i] if i < len(user_turn_rewards) else 0.0
                print(f"[Segment Reward Debug] Agent segment {i}: relative tokens {seg['start']}-{seg['end']}, reward: {reward}")
        
        # If no agent segments remain after filtering, fallback to last token
        if len(all_agent_segments) == 0:
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
            return

        if self.reward_allocation == "last_token":
            # Assign reward only to the last token of the last agent segment
            last_segment = all_agent_segments[-1]
            last_token_pos = min(last_segment["end"], valid_response_length - 1)
            # Use the last user_turn_reward (final step gets largest reward)
            final_reward = user_turn_rewards[-1] if user_turn_rewards else reward_to_distribute
            reward_tensor[batch_idx, last_token_pos] = final_reward
            
        elif self.reward_allocation == "uniform_positive":
            # Distribute rewards across all agent segments using user_turn_rewards
            for response_idx, seg in enumerate(all_agent_segments):
                if response_idx < len(user_turn_rewards):
                    segment_reward = user_turn_rewards[response_idx]
                    start = seg["start"]
                    end = min(seg["end"], valid_response_length - 1)
                    if start < valid_response_length:
                        segment_len = end - start + 1
                        if segment_len > 0:
                            reward_tensor[batch_idx, start:end+1] = segment_reward
                
        elif self.reward_allocation == "discounted":
            # Distribute reward with temporal discounting, using user_turn_rewards
            gamma = self.gamma
            
            # Iterate segments in reverse order (last to first)
            for response_idx, seg in enumerate(reversed(all_agent_segments)):
                # Get the original index (not reversed)
                original_idx = len(all_agent_segments) - 1 - response_idx
                
                if original_idx < len(user_turn_rewards):
                    # Apply discount: later segments get more reward
                    discount_factor = gamma ** response_idx
                    segment_reward = user_turn_rewards[original_idx] * discount_factor
                    
                    start = seg["start"]
                    end = min(seg["end"], valid_response_length - 1)
                    
                    if start < valid_response_length:
                        segment_len = end - start + 1
                        # Safety check: skip empty segments
                        if segment_len <= 0:
                            if hasattr(self, 'num_examine') and self.num_examine > 0:
                                print(f"[Debug] Skipping empty segment: start={start}, end={end}, segment_len={segment_len}")
                            continue
                        reward_tensor[batch_idx, start:end+1] = segment_reward
            
        else:
            # Unknown strategy, fallback to last token
            reward_tensor[batch_idx, valid_response_length - 1] = reward_to_distribute
