#!/bin/bash
#SBATCH --job-name=tog_2multiwiki
#SBATCH --output=./token/llama/2multiwiki.out  # 输出文件，%A 是作业ID，%a 是数组索引（如果使用了数组作业）
#SBATCH --error=./token/llama/2multiwiki.err    # 错误文件
#SBATCH --nodes=1                  # 请求的节点数
#SBATCH --ntasks=1                # 请求的任务数（通常与节点数相匹配，除非使用了多核任务）
#SBATCH --cpus-per-task=8        # 每个任务请求的CPU数
#SBATCH --gres=gpu:2
#SBATCH --partition=A40
#SBATCH --exclude=comput06,comput01,comput02

# 这里是你的作业命令
echo "Running job on node: $(hostname)"
# 替换下面的命令为你的实际作业命令
cd ../
cd ToG
/share/home/ncu_418000240001/test/long/bin/python -u main_wiki.py \
  --dataset 2multiwiki \
  --max_length 256 \
  --temperature_exploration 0.4 \
  --temperature_reasoning 0 \
  --width 3 \
  --depth 3 \
  --remove_unnecessary_rel True \
  --LLM_type phi \
  --opeani_api_keys sk-xxxx \
  --num_retain_entity 5 \
  --prune_tools llm
date
hostname
