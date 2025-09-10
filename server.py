pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(model_path="/path/to/model.gguf")
output = llm("Q: What is DeepSeek? A:")
print(output)


pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/deepseek \
  --port 38000

source /workspace/miniconda3/bin/activate
conda init

https://<your-pod-id>-<port>-<region>.runpodapis.com

python -m ipykernel install --user --name ktransformer --display-name "Python (ktransformer)"

python -m vllm.entrypoints.openai.api_server  --model /workspace/qwent3_8B   --port 38000 --max-model-len 10480  --gpu-memory-utilization 0.90  


from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_dir="/workspace/Qwen_1_5B")
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", local_dir="/workspace/qwent3_8B")
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", local_dir="/workspace/qwent_7B")
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", local_dir="/workspace/qwen_32B")
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_dir="/workspace/Qwen_1_5B")


snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", local_dir="/workspace/deepseek")
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", local_dir="/workspace/llama_8b")


DeepSeek-R1-Distill-Qwen-14B


ktransformers --model_path /workspace/llama_8b --port 8000 --web True

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

pip install vllm nvidia-cudnn-cu12==8.9.2.26 nvidia-cufft-cu12==11.0.2.54

pip install ipykernel
conda activate base  # or
pip install vllm
python -c "import torch; print(torch.version.cuda); print(torch.__version__)"
nvcc --version
nvidia-smi


python -m ipykernel install --user --name=llm --display-name="Python (llm)"


curl https://1g40rvgkpqqbur-8000.proxy.runpod.net/v1/models


curl https://1g40rvgkpqqbur-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-anything" \
  -d '{
    "model": "/workspace/qwent3_8B",
    "messages": [
      { "role": "user", "content": "where is new york?" }
    ]
}'


curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-anything" \
  -d '{
    "model": "/workspace/deepseek",
    "messages": [
      { "role": "user", "content": "where is new york?" }
    ]
}'


conda create 

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/qwent_7B \
    --host 0.0.0.0 \
    --port 8002 \
    --max-model-len 4768 \
    --gpu-memory-utilization 0.92 \
    --served-model-name agent-llm \
    --tensor-parallel-size 2 \
  

ktransformers --model_path /workspace/DeepSeek-V2-Lite --gguf_path /workspace/DeepSeek-V2-Lite-Chat-GGUF --port 8000

curl https://b6dgqpsi8k09p9-8002.proxy.runpod.net/v1/model
curl -X POST https://hwhr7kt1hve3by-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-Coder-V2-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "stream": false,
    "max_tokens": 50
  }' -v'
  '


ssh -i ~/.ssh/id_ed25519 -p 10211 -L 35678:localhost:35678 root@203.57.40.104

python -m  openmanus_rl.agentgym.evaluation.vllm_eval_tools_todo --enable-debug --model_path /workspace/qwent_7B


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /workspace/qwent_7B --host 0.0.0.0  --port 8002  --max-model-len 8768  --gpu-memory-utilization 0.96  --served-model-name agent-llm  --tensor-parallel-size 2


torchrun --nproc_per_node=1 ./verl/tests/workers/rollout/test_openmanus_async_rollout.py

ssh -N -f root@203.57.40.106 -p 10127 -L 8000:localhost:8000 -i ~/.ssh/id_ed25519
ssh -N -f root@203.57.40.188 -p 10190 -L 8000:localhost:8000 -i ~/.ssh/id_ed25519



“Find the documents (K) most relevant to my question (Q), and return their content (V).”



wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh


conda create -n openrl python=3.10
conda activate openrl


pip3 install torch torchvision
pip install -e .[vllm]
pip3 install flash-attn --no-build-isolation
pip install wandb
pip install debugpy


python3 -m vllm.entrypoints.openai.api_server \
  --model ./merged_model_fp16 \
  --dtype float16 \
  --port 38000 \
  --gpu-memory-utilization 0.9
ray start --head --port=6379 

ray start --head \
    --port=6380 \
    --dashboard-port=8265 \
    --dashboard-host=0.0.0.0 \
    --node-ip-address=127.0.0.1 \
    --verbose
    
    --node-ip-address=66.244.106.221 \


ray start --address='127.0.0.1:6379' --node-ip-address='127.0.0.1' --port='6479'

git config

lsof -i -P -n | grep LISTEN

tail -n 100 -f /tmp/ray/session_latest/logs/*.log

RAY_USE_MULTIPROCESSING_CPU_COUNT=1
tail -n 100 -f /tmp/ray/session_latest/logs/monitor.log

tail -n 100 /tmp/ray/session_2025-07-19_01-44-20_765712_4936/logs/log_monitor.2.err

tail -n 100 /tmp/ray/session_latest/logs/*.log

bazel build //:ray_pkg    
SKIP_BAZEL_BUILD=1 RAY_INSTALL_CPP=0 pip install -e .

git remote add origin https://$GH_TOKEN@github.com/freedomkwok/OpenRL.git
git remote set-url origin https://$GH_TOKEN@github.com/freedomkwok/OpenRL.git

git remote add origin https://github.com/freedomkwok/verl_test.git
git remote set-url origin https://github.com/freedomkwok/verl_test.git

pip install -e '.[sglang,vllm,gpu]'

cd OpenRL && git remote set-url origin https://$GH_TOKEN@github.com/freedomkwok/OpenRL.git && git reset --hard origin/main  && git pull

mkdir -p /workspace/OpenRL2 && mkdir -p /data/verl_checkpoints && mkdir -p /data/models \
mkdir -p /output/logs/ && cd /workspace/OpenRL2 && git init && git remote add origin https://$GH_TOKEN@github.com/freedomkwok/OpenRL.git && git fetch --all  && git checkout -b main_09_01 origin/main_09_01 && git pull && git submodule update --init --recursive \
&& cd /workspace/OpenRL2 && conda create -n openrl python=3.10 -y && conda activate openrl && cd /workspace/OpenRL2  && pip install -e . && pip uninstall verl -y && cd /workspace/OpenRL2/verl && pip install -e '.[sglang,gpu]' && pip install flash-attn --no-build-isolation


mkdir -p /data/models && mkdir -p /workspace/OpenRL2 && cd /workspace/OpenRL2 \
git init && git remote add origin https://github.com/freedomkwok/verl_test.git && git checkout -b main origin/main && git pull && \ 
conda create -n openrl2 python=3.10 -y && conda activate openrl2 && pip install -e '.[sglang]' && pip install flash-attn --no-build-isolation


conda create -n openrl2 python=3.10 -y && conda activate openrl2 &&  && pip install -e . && pip uninstall verl -y && cd /workspace/OpenRL2/verl && pip install -e '.[sglang,gpu]' && pip install flash-attn --no-build-isolation

CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_P2P_DISABLE=0 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONUNBUFFERED=1 DEBUGGY_LOCAL=True SGLANG_LOG_LEVEL=INFO \
torchrun --nproc_per_node=1 tests/workers/rollout/test_openmanus_async_rollout.py

CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_P2P_DISABLE=0 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONUNBUFFERED=1 DEBUGGY_LOCAL=True SGLANG_LOG_LEVEL=INFO \
torchrun --nproc_per_node=1 tests/workers/rollout/test_sglang_async_rollout_w_interaction.py


cd /workspace/OpenRL && conda create -n agent_server python=3.11 -y && conda activate agent_server && cd /workspace/OpenRL/openmanus_rl/agentgym/agentenv-gmail_calendar && pip install -e . && agentenv_gmail_calendar

pip install "sglang[srt,openai]==0.4.6.post5"

git branch --set-upstream-to=origin/main main
git reset --hard origin/main 
git checkout -b main origin/main

echo "" >> ~/.bashrc
echo "export LOCAL_IP=\$(hostname -I | awk '{print \$1}')" >> ~/.bashrc
source ~/.bashrc

rm -rf /tmp/*
rm -rf ~/.cache/*
rm -rf /var/tmp/*
rm -rf ~/.cache/pip

mkdir -p /data/models/QWEN1_5B_MOD1/
mkdir -p /data/logs/
mkdir -p /data/verl_checkpoints/

ssh -N -f -p 41042 -L 41042:localhost:41042 root@143.55.45.86

ps -ef | grep "bash train_grpo_gmail.sh" | grep -v grep | awk '{print $2}' | xargs -r kill -9


torch.where(initial_prompt_ids[0] != self.tokenizer.pad_token_id)[0][0].item()
torch.where(current_input_ids[0] != self.tokenizer.pad_token_id)[0][0].item()
torch.where(padded_gen_input_proto.batch["input_ids"][0] != self.tokenizer.pad_token_id)[0][0].item()

torch.where(temp2.batch["responses"][0] != self.tokenizer.pad_token_id)[0][0].item()

torch.where(padded_gen_input_proto.batch["input_ids"][0] != self.tokenizer.pad_token_id)[0][0].item()


torch.where(padded_gen_input_proto.batch["attention_mask"][0] != 0)[0][0].item()

torch.where(padded_gen_input_proto.batch["position_ids"][0] != 0)[0][0].item()

torch.where(data_item.batch["input_ids"] != self.tokenizer.pad_token_id)[0][0].item()

torch.where(micro_batch["input_ids"] != tokenizer.pad_token_id)[0][0].item()

torch.where((full_input_ids[0] * prompt_masks[0]) != 0)[0]

torch.where(prompt_masks[0] != 0)[0][0].item()

#from micro_forward
tokenizer.decode(torch.argmax(output.logits[0], dim=-1)[-100:], skip_special_tokens=True)
tokenizer.decode(micro_batch['responses'][0], skip_special_tokens=True)

self.tokenizer.decode((full_input_ids[0] * prompt_masks[0]), skip_special_tokens=True)

prompt_text = (full_input_ids[0] * prompt_masks[0])
torch.where(prompt_text != 0, prompt_text, torch.tensor(self.tokenizer.pad_token_id))
data_item.batch['prompt_masks'].sum().item()

pred_token_ids = torch.argmax(output.logits, dim=-1)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/QWEN1_5B_0815_A", trust_remote_code=True)
tokenizer.decode(pred_token_ids[0].tolist(), skip_special_tokens=True)
tokenizer.decode(micro_batch['responses'][0].tolist() , skip_special_tokens=True)

tokenizer.decode(log_prob[0].tolist(), skip_special_tokens=True)

torch.sum(kl[0])

print("Actor module in train mode?", self.actor_module.training)
import torch.nn as nn
train_count = 0
eval_count = 0
for m in self.actor_module.modules():
    if hasattr(m, "training"):
        if m.training:
            train_count += 1
        else:
            eval_count += 1

print(f"Submodules: {train_count} in train mode, {eval_count} in eval mode")

dropouts_train, dropouts_eval = 0, 0
for m in self.actor_module.modules():
    if isinstance(m, nn.Dropout) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        if m.training:
            dropouts_train += 1
        else:
            dropouts_eval += 1

print(f"Dropouts/BN: {dropouts_train} train, {dropouts_eval} eval")

total_params = sum(p.numel() for p in actor_module.parameters())
print(f"Actor module total parameters: {total_params/1e9:.2f}B")

import re, torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd

def no_dropout_summary(model):
    counts = {"Dropout":0, "BatchNorm":0, "LayerNorm":0, "RMSNorm_like":0}
    for m in model.modules():
        if isinstance(m, _DropoutNd): counts["Dropout"]+=1
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): counts["BatchNorm"]+=1
        if isinstance(m, nn.LayerNorm): counts["LayerNorm"]+=1
        if re.search("rmsnorm", type(m).__name__, re.I): counts["RMSNorm_like"]+=1
    print(counts)


    === Module type counts (top 50) ===
Linear                           197
Qwen2RMSNorm                     57
Qwen2RotaryEmbedding             29
Qwen2DecoderLayer                28
Qwen2SdpaAttention               28
Qwen2MLP                         28
SiLU                             28
Qwen2ForCausalLM                 1
Qwen2Model                       1
Embedding                        1
ModuleList                       1