# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import requests
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from requests.exceptions import RequestException

from verl.utils.reward_score import gmail

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GmailInteraction(BaseInteraction):
    """A Gmail interaction that communicates directly with the Gmail server.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        
        # Gmail server configuration
        self.env_server_base = config.get("env_server_base", "http://127.0.0.1")
        self.env_server_port = config.get("env_server_port", 8000)
        self.timeout = config.get("timeout", 600)

        self.max_steps = config.get("max_steps", 10)
        # Environment state
        self._env_id = None
        self._job_id = 1

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request to Gmail server."""
        if self._env_id is not None:
            data["env_idx"] = self._env_id
        
        response = requests.post(
            f"{self.env_server_base}:{self.env_server_port}/{path}",
            json=data,
            timeout=self.timeout,
        )
        
        if response.status_code != 200:
            raise RequestException(f"Failed to communicate with Gmail server: {response.status_code} - {response.text}")
        
        return response.json()

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        if kwargs.get("job_id") is not None:
            self._job_id = kwargs.get("job_id")
        if kwargs.get("ground_truth") is not None:
            ground_truth = kwargs.get("ground_truth")
        try:
            # Create environment if not exists
            if self._env_id is None:
                create_result = self._post("create", {"env_idx": 0, "job_id": self._job_id})
                self._env_id = create_result["env_idx"]
            
            # Reset environment for new interaction
            reset_result = self._post("reset", {"env_idx": self._env_id, "job_id": self._job_id})
            
            self._instance_dict[instance_id] = {
                "response": "",
                "ground_truth": ground_truth,
                "reward": 0.0,
                "step": 0,
                "observation": reset_result.get("observation", ""),
                "owner_tag": reset_result.get("owner_tag", ""),
                "owner_email": reset_result.get("owner_email", ""),
                "owner_name": reset_result.get("owner_name", ""),
            }
            
            logger.info(f"Gmail interaction started for instance {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to start Gmail interaction: {e}")
            self._instance_dict[instance_id] = {
                "response": "",
                "ground_truth": ground_truth,
                "reward": 0.0,
                "step": 0,
                "observation": "",
                "owner_tag": "",
                "owner_email": "",
                "owner_name": "",
            }
        
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, dict]:
        # Get the latest assistant message (which might contain tool calls or actions)
        latest_message = messages[-1] if messages else {}
        
        # Check if this is a tool call response
        if latest_message.get("role") == "assistant" and latest_message.get("tool_calls"):
            # Handle tool calls - execute them and return results
            tool_results = await self._execute_tool_calls(latest_message["tool_calls"])
            return False, tool_results, 0.0, {"tool_calls": latest_message["tool_calls"]}
        
        # Handle environment stepping if we have an environment
        if instance_id in self._instance_dict:
            instance_data = self._instance_dict[instance_id]

            # Extract action from the latest message
            action = latest_message.get("content", "") if latest_message.get("role") == "assistant" else ""
            
            if action and instance_data["step"] < 10:  # Max 10 steps
                try:
                    # Step the environment
                    step_result = self._post("step", {"action": action})
                    
                    # Update instance data
                    instance_data["observation"] = step_result.get("observation", "")
                    instance_data["step"] += 1
                    reward = float(step_result.get("reward", 0.0))
                    instance_data["reward"] += reward
                    done = step_result.get("done", False)
                    
                    response = instance_data["observation"]
                    # Check if episode is done
                    if done:
                        # response = f"Episode completed! Total reward: {instance_data['reward']}"
                        should_terminate_sequence = True
                    else:
                        # response = f"Step {instance_data['step']}: Action executed. Reward: {reward}. Observation: {instance_data['observation']}"
                        should_terminate_sequence = False
                    
                    return should_terminate_sequence, response, instance_data["reward"], {}
                    
                except Exception as e:
                    logger.error(f"Failed to step Gmail environment: {e}")
                    return True, f"Environment error: {str(e)}", 0.0, {}
        
        # Fallback to original GSM8K logic for backward compatibility
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "user":
                content = item.get("content")
                break

        if content and content.startswith("#### "):
            self._instance_dict[instance_id]["response"] = content
        else:
            self._instance_dict[instance_id]["response"] = "#### " + (content or "")

        reward = await self.calculate_score(instance_id)
        if reward == 1.0:
            response = "Your response is correct!"
            should_terminate_sequence = True
        else:
            response = "Your response is incorrect! You need to reflect on your answer and try again."
            should_terminate_sequence = False

        return should_terminate_sequence, response, reward, {}

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute tool calls and return results as a formatted string."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args = tool_call.get("function", {}).get("arguments", "{}")
            
            try:
                # For now, just return a mock result
                # In the future, you can implement actual tool execution here
                result = f"Tool {tool_name} executed with args: {tool_args}"
                results.append(f"Tool {tool_name}: {result}")
            except Exception as e:
                results.append(f"Tool {tool_name} failed: {str(e)}")
        
        return "\n".join(results)

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return gmail.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            self._instance_dict[instance_id]["extra_info"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
        
        # Optionally close the environment
        if self._env_id is not None:
            try:
                self._post("close", {"env_idx": self._env_id})
            except Exception as e:
                logger.warning(f"Failed to close Gmail environment: {e}")