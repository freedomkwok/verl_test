#!/bin/bash
timer=${1:-2}
watch -n $timer nvidia-smi
