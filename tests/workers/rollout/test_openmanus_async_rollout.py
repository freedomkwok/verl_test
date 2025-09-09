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
"""
usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=2 $(which pytest) \
    -s test_openmanus_async_rollout.py
"""

import numpy as np
import torch
from tensordict import TensorDict
import torch.distributed as dist
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from torch.distributed.device_mesh import init_device_mesh
# from verl.workers.rollout.openmanus_rollout import OpenManusRollout
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


def test_async_openmanus_rollout():
    """Test OpenManusRollout with Gmail environment interaction."""

    max_prompt_length = 3000
    max_response_length = 1500
    dtype = "bfloat16"
    tensor_parallel_size = 1
    local_model_path = "/data/models/QWEN1_5B_0815_A"

    initialize_global_process_group()
    clean_torchelastic_env()

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = [
        [{"role": "user", "content": prompt, "tool_calls": None}]
        for prompt in [
            "Check my emails and schedule a meeting with John for tomorrow at 2pm"
        ]
    ]
    


    # OpenManus environment configuration
    env_configs = [
        {
            "env_name": "gmail",
            "env_ports": [8000],
            "env_server_base": "http://localhost",
            "max_turns": 10,
            "max_prompt_length": max_prompt_length,
        }
    ]
    
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in preencode_prompts
    ]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)
    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)
    # Create a temporary interaction config file for testing
    import tempfile
    from omegaconf import OmegaConf

    rollout_config = OmegaConf.create({
        # Basic rollout configs (from utils_sglang.py)
        "name": "gmail",
        "mode": "async",
        "load_format": "dummy",
        "enforce_eager": False,
        "free_cache_engine": True,
        
        # Model and distributed configs
        "dtype": dtype,
        "tensor_model_parallel_size": 1,   # IMPORTANT
        "nnodes": 1,                       # make it explicit
        "node_rank": 0,
        "tensor_model_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": 0.5,
        "ignore_eos": False,
        "max_num_batched_tokens": 8192,

                # Required length configs
        "max_response_length": max_response_length,
        "max_prompt_length": max_prompt_length,
        "prompt_length": max_prompt_length,
        "response_length": max_response_length,
        "max_model_len": max_prompt_length + max_response_length,  # Required by SGLangRollout
        
        # Multi-turn configs
        "multi_turn": {
            "enable": True,
            "max_assistant_turns": 3,
            "max_user_turns": 2,
            "tool_config_path": None,
            "interaction_config_path": "examples/sglang_multiturn/config/interaction_config/gmail_interaction_config.yaml",
            "use_inference_chat_template": False,
            "tokenization_sanity_check_mode": "off",
        },

        # Sampling configs
        "calculate_log_probs": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "stop": [],
        "n": 1,
        "do_sample": True,
    })

    # rollout_config = get_rollout_config(
    #     max_response_length, max_prompt_length, dtype, tensor_parallel_size, None, interaction_config_path
    # )
    # rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config, dataclass_type=RolloutConfig)

    model_config = HFModelConfig(path=local_model_path)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=("dp", "tp", "pp"))

    rollout = SGLangRollout(
        config=rollout_config,
        model_config=model_config,
        device_mesh=device_mesh,
    )

    prompt_dict = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    print(f"preprocessed {input_ids.shape=}")

    messages = np.asarray(preencode_prompts)
    interaction_kwargs = [
        {"name": "gmail", "query": "Check my emails and schedule a meeting with John for tomorrow at 2pm", "ground_truth": "Meeting scheduled with John for tomorrow at 2pm"},
    ]
    prompts = DataProto(
        batch=prompt_dict,
        non_tensor_batch={"raw_prompt": messages, "interaction_kwargs": np.asarray(interaction_kwargs)},
    )

    prompts.meta_info.update(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    # log_gpu_memory_usage("Before generating sequences", logger=None)
    output = rollout.generate_sequences(prompts=prompts)
    print(f"generated {output.batch['responses'].shape=}")
    # log_gpu_memory_usage("After generating sequences", logger=None)

    openmanus_output = output.to("cpu")
    openmanus_response_tokens = tokenizer.batch_decode(openmanus_output.batch["responses"],
                                                    skip_special_tokens=True,       # removes <pad>, <eos>, etc.
                                                    clean_up_tokenization_spaces=True)

    print(f"hf response: {hf_response_tokens}")
    print(f"openmanus response: {openmanus_response_tokens}")
      
    print("OpenManus Rollout Test Passed!")

    # Clean up temporary config file
    import os

    os.unlink(interaction_config_path)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()

    # test_openmanus_rollout_initialization()
    test_async_openmanus_rollout()  # Commented out as it requires actual Gmail servers
