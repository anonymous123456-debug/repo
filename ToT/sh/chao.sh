#!/bin/bash
#SBATCH --job-name=squad  # 作业名称
#SBATCH --output=./token/smol/squad.out  # 输出文件，%A 是作业ID，%a 是数组索引（如果使用了数组作业）
#SBATCH --error=./token/smol/squad.err    # 错误文件
#SBATCH --nodes=1                  # 请求的节点数
#SBATCH --ntasks=1                # 请求的任务数（通常与节点数相匹配，除非使用了多核任务）
#SBATCH --cpus-per-task=8        # 每个任务请求的CPU数
#SBATCH --gres=gpu:2
#SBATCH --partition=A40 
#SBATCH --exclude=comput06,comput01,comput02

# 这里是你的作业命令
echo "Running job on node: $(hostname)"
# 替换下面的命令为你的实际作业命令
cd ../../ 
source .bashrc
conda activate long
cd tt2
/share/home/ncu_418000240001/test/long/bin/python -u run.py --task squad --method_generate propose  --method_evaluate value --method_select greedy --n_generate_sample 20 --n_evaluate_sample 3 --n_select_sample 3 --step 3
# 例如，运行一个简单的命令来打印当前节点的名称和日期 --nodelist=comput07
date
hostname
