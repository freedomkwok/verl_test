#!/bin/bash


ps -ef | grep "bash train_grpo_gmail.sh" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Kill all test processes (torchrun and test scripts)

ps aux | grep "test_openmanus_async_rollout" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "test" | grep -v grep | awk '{print $2}' | xargs -r kill -9

