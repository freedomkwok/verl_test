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

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from verl.interactions.gmail_interaction import GmailInteraction


class TestGmailInteraction:
    """Test suite for GmailInteraction class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = {
            "env_server_base": "http://127.0.0.1",
            "env_server_port": 8000,
            "timeout": 30
        }
        self.gmail_interaction = GmailInteraction(self.config)

    def test_init(self):
        """Test GmailInteraction initialization."""
        assert self.gmail_interaction.env_server_base == "http://127.0.0.1"
        assert self.gmail_interaction.env_server_port == 8000
        assert self.gmail_interaction.timeout == 30
        assert self.gmail_interaction._env_id is None
        assert self.gmail_interaction._job_id == 1
        assert isinstance(self.gmail_interaction._instance_dict, dict)

    @patch('requests.post')
    async def test_start_interaction_success(self, mock_post):
        """Test successful interaction start."""
        # Mock successful server responses
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {
                "observation": "Gmail environment ready",
                "owner_tag": "test_user",
                "owner_email": "test@example.com",
                "owner_name": "Test User"
            })
        ]

        instance_id = await self.gmail_interaction.start_interaction(
            ground_truth="Test ground truth"
        )

        # Verify instance was created
        assert instance_id in self.gmail_interaction._instance_dict
        instance_data = self.gmail_interaction._instance_dict[instance_id]
        assert instance_data["ground_truth"] == "Test ground truth"
        assert instance_data["observation"] == "Gmail environment ready"
        assert instance_data["owner_email"] == "test@example.com"
        assert instance_data["step"] == 0
        assert instance_data["reward"] == 0.0

        # Verify server calls
        assert mock_post.call_count == 2
        mock_post.assert_any_call(
            "http://127.0.0.1:8000/create",
            json={"env_idx": 1, "job_id": 1},
            timeout=30
        )

    @patch('requests.post')
    async def test_start_interaction_with_custom_instance_id(self, mock_post):
        """Test interaction start with custom instance ID."""
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"})
        ]

        custom_id = "custom_test_id"
        instance_id = await self.gmail_interaction.start_interaction(
            instance_id=custom_id
        )

        assert instance_id == custom_id
        assert custom_id in self.gmail_interaction._instance_dict

    @patch('requests.post')
    async def test_start_interaction_server_error(self, mock_post):
        """Test interaction start with server error."""
        mock_post.side_effect = Exception("Connection failed")

        instance_id = await self.gmail_interaction.start_interaction()

        # Should still create instance with default values
        assert instance_id in self.gmail_interaction._instance_dict
        instance_data = self.gmail_interaction._instance_dict[instance_id]
        assert instance_data["observation"] == ""
        assert instance_data["reward"] == 0.0

    @patch('requests.post')
    async def test_generate_response_tool_calls(self, mock_post):
        """Test response generation with tool calls."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"})
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        # Test tool call handling
        messages = [
            {"role": "user", "content": "Check my emails"},
            {
                "role": "assistant",
                "content": "I'll check your emails for you.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "gmail_search",
                            "arguments": '{"query": "in:inbox"}'
                        }
                    }
                ]
            }
        ]

        should_terminate, response, reward, metadata = await self.gmail_interaction.generate_response(
            instance_id, messages
        )

        assert should_terminate is False
        assert "Tool gmail_search executed" in response
        assert reward == 0.0
        assert "tool_calls" in metadata

    @patch('requests.post')
    async def test_generate_response_environment_step(self, mock_post):
        """Test response generation with environment stepping."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"}),
            Mock(status_code=200, json=lambda: {
                "observation": "Action executed successfully",
                "reward": 0.5,
                "done": False
            })
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        # Test environment stepping
        messages = [
            {"role": "user", "content": "Send an email"},
            {"role": "assistant", "content": "I'll send an email for you."}
        ]

        should_terminate, response, reward, metadata = await self.gmail_interaction.generate_response(
            instance_id, messages
        )

        assert should_terminate is False
        assert "Step 1: Action executed" in response
        assert "Reward: 0.5" in response
        assert reward == 0.5

        # Verify instance data was updated
        instance_data = self.gmail_interaction._instance_dict[instance_id]
        assert instance_data["step"] == 1
        assert instance_data["reward"] == 0.5

    @patch('requests.post')
    async def test_generate_response_episode_completion(self, mock_post):
        """Test response generation when episode is completed."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"}),
            Mock(status_code=200, json=lambda: {
                "observation": "Task completed",
                "reward": 1.0,
                "done": True
            })
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        messages = [
            {"role": "user", "content": "Complete the task"},
            {"role": "assistant", "content": "Task completed."}
        ]

        should_terminate, response, reward, metadata = await self.gmail_interaction.generate_response(
            instance_id, messages
        )

        assert should_terminate is True
        assert "Episode completed!" in response
        assert "Total reward: 1.0" in response
        assert reward == 1.0

    @patch('requests.post')
    async def test_generate_response_max_steps(self, mock_post):
        """Test response generation when max steps reached."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"})
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        # Set step to max (10)
        self.gmail_interaction._instance_dict[instance_id]["step"] = 10

        messages = [
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Trying to continue"}
        ]

        should_terminate, response, reward, metadata = await self.gmail_interaction.generate_response(
            instance_id, messages
        )

        # Should fall back to GSM8K logic since step >= 10
        assert "Your response is" in response

    @patch('requests.post')
    async def test_generate_response_environment_error(self, mock_post):
        """Test response generation with environment error."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"}),
            Exception("Environment error")
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "I'll do it."}
        ]

        should_terminate, response, reward, metadata = await self.gmail_interaction.generate_response(
            instance_id, messages
        )

        assert should_terminate is True
        assert "Environment error" in response
        assert reward == 0.0

    async def test_calculate_score(self):
        """Test score calculation."""
        instance_id = "test_instance"
        self.gmail_interaction._instance_dict[instance_id] = {
            "response": "#### 42",
            "ground_truth": "42",
            "reward": 0.0,
            "step": 0,
            "observation": "",
            "owner_tag": "",
            "owner_email": "",
            "owner_name": "",
        }

        with patch('verl.utils.reward_score.gmail.compute_score', return_value=1.0) as mock_compute:
            score = await self.gmail_interaction.calculate_score(instance_id)
            
            assert score == 1.0
            mock_compute.assert_called_once_with(
                "#### 42",
                "42",
                method="flexible",
                format_score=0.0,
                score=1.0
            )

    @patch('requests.post')
    async def test_finalize_interaction(self, mock_post):
        """Test interaction finalization."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"}),
            Mock(status_code=200, json=lambda: {"status": "closed"})
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        # Verify instance exists
        assert instance_id in self.gmail_interaction._instance_dict

        # Finalize interaction
        await self.gmail_interaction.finalize_interaction(instance_id)

        # Verify instance was removed
        assert instance_id not in self.gmail_interaction._instance_dict

        # Verify close was called
        mock_post.assert_called_with(
            "http://127.0.0.1:8000/close",
            json={"env_idx": 1},
            timeout=30
        )

    @patch('requests.post')
    async def test_finalize_interaction_close_error(self, mock_post):
        """Test interaction finalization with close error."""
        # Start an interaction first
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda: {"env_idx": 1}),
            Mock(status_code=200, json=lambda: {"observation": "Ready"}),
            Exception("Close failed")
        ]
        instance_id = await self.gmail_interaction.start_interaction()

        # Should not raise exception even if close fails
        await self.gmail_interaction.finalize_interaction(instance_id)

        # Verify instance was still removed
        assert instance_id not in self.gmail_interaction._instance_dict

    async def test_execute_tool_calls(self):
        """Test tool call execution."""
        tool_calls = [
            {
                "function": {
                    "name": "gmail_search",
                    "arguments": '{"query": "in:inbox"}'
                }
            },
            {
                "function": {
                    "name": "gmail_send",
                    "arguments": '{"to": "test@example.com", "subject": "Test"}'
                }
            }
        ]

        result = await self.gmail_interaction._execute_tool_calls(tool_calls)

        assert "Tool gmail_search executed" in result
        assert "Tool gmail_send executed" in result
        assert "in:inbox" in result
        assert "test@example.com" in result

    async def test_execute_tool_calls_with_error(self):
        """Test tool call execution with error."""
        tool_calls = [
            {
                "function": {
                    "name": "invalid_tool",
                    "arguments": "{}"
                }
            }
        ]

        result = await self.gmail_interaction._execute_tool_calls(tool_calls)

        assert "Tool invalid_tool executed" in result

    def test_post_method(self):
        """Test the _post method."""
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: {"result": "success"})

            result = self.gmail_interaction._post("test", {"data": "test"})

            assert result == {"result": "success"}
            mock_post.assert_called_once_with(
                "http://127.0.0.1:8000/test",
                json={"data": "test"},
                timeout=30
            )

    def test_post_method_with_env_id(self):
        """Test the _post method with environment ID."""
        self.gmail_interaction._env_id = 123

        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: {"result": "success"})

            result = self.gmail_interaction._post("test", {"data": "test"})

            assert result == {"result": "success"}
            mock_post.assert_called_once_with(
                "http://127.0.0.1:8000/test",
                json={"data": "test", "env_idx": 123},
                timeout=30
            )

    def test_post_method_error(self):
        """Test the _post method with error response."""
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=500, text="Internal Server Error")

            with pytest.raises(Exception, match="Failed to communicate with Gmail server"):
                self.gmail_interaction._post("test", {"data": "test"})


# Run the tests
if __name__ == "__main__":
    # Example of running a specific test
    async def run_single_test():
        test_instance = TestGmailInteraction()
        test_instance.setup_method()
        
        # Test initialization
        test_instance.test_init()
        print("✓ Initialization test passed")
        
        # Test with mocked server
        with patch('requests.post') as mock_post:
            mock_post.side_effect = [
                Mock(status_code=200, json=lambda: {"env_idx": 1}),
                Mock(status_code=200, json=lambda: {"observation": "Ready"})
            ]
            instance_id = await test_instance.gmail_interaction.start_interaction()
            print(f"✓ Start interaction test passed - Instance ID: {instance_id}")
        
        print("All tests completed successfully!")

    # Run the test
    asyncio.run(run_single_test())
