#!/bin/bash
#SBATCH --job-name=medqa
#SBATCH --output=./token/phi/rag_medqa.out  # 输出文件，%A 是作业ID，%a 是数组索引（如果使用了数组作业）
#SBATCH --error=./token/phi/rag_medqa.err    # 错误文件
#SBATCH --nodes=1                  # 请求的节点数
#SBATCH --ntasks=1                # 请求的任务数（通常与节点数相匹配，除非使用了多核任务）
#SBATCH --cpus-per-task=8        # 每个任务请求的CPU数
#SBATCH --gres=gpu:2
#SBATCH --partition=A40
#SBATCH --exclude=comput06,comput01,comput02
#SBATCH --nodelist=comput11


# 这里是你的作业命令
echo "Running job on node: $(hostname)"
# 替换下面的命令为你的实际作业命令
cd ../
cd ../
source .bashrc
conda activate rag
cd LongRAG-main/src
/share/home/ncu_418000240001/test/long/bin/python -u  main.py --dataset medqa --model phi-3.8b --rb --r_path ../data/corpus/processed/200_2_2
date
hostname
