#!/bin/bash
#SBATCH --job-name=rust_run             # 作业名
#SBATCH --nodes=1                      # 使用的节点数
#SBATCH --ntasks-per-node=24            # 每个节点的任务数
#SBATCH --output=rust_run_%j.out        # 输出文件名，%j 代表作业ID
#SBATCH --error=rust_run_%j.err         # 错误文件名，%j 代表作业ID
#SBATCH --time=12:00:00                # 预计作业运行时间
#SBATCH --mem=256G                     # 请求每个节点的内存容量
#SBATCH --mail-type=ALL                # 结束/失败时发送邮件
#SBATCH --mail-user=zihanwu7           # 设置接收邮件的邮箱



export MYHOME=/home/zihanwu7
export BASE_DIR=$MYHOME/fast_cocluster

# 加载环境变量
source $MYHOME/.cargo/env

# 进入工作目录
cd $BASE_DIR

# 执行程序
cargo run --release
