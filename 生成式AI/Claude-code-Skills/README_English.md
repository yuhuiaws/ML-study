# GPU Infra as Claude Code Skills

A collection of Claude Code skills for deploying and training large language models (primarily DeepSeek-V3/V4, Kimi2.5, Qwen3, mimo-v2-flash) on AWS GPU infrastructure using SGLang, VLLM, and Megatron-LM. These skills automate cluster provisioning, model deployment, model training, benchmarking, and PD (Prefill-Decode) disaggregation configurations.

## Skills Overview

### SGLang Deployment Skills

| Skill | Description |
|-------|-------------|
| `sglang-deepseek-non-PD.skill` | Non-PD deployment for DeepSeek-V3 FP8. Covers single-node and 2-node deployments. 2-node deployment experiments include both NCCL-based and UCCL-EP-based approaches (UCCL-EP used for comparative benchmarking). |
| `sglang-deepseek-1p1d.skill` | 1P1D (1 Prefill + 1 Decode) deployment using NIXL KV transfer engine, comparing two backends: libfabric and UCX. |
| `sglang-2p2d-nccl-nixl.skill` | 2P2D deployment across 4 GPU instances. Communication between 2 prefill nodes uses NCCL; communication between 2 decode nodes also uses NCCL. |
| `sglang-2p2d-ucclep-nixl.skill` | 2P2D deployment with two variants: (1) Independent prefill nodes (no inter-P communication) + UCCL-EP all2all between decode nodes; (2) UCCL-EP all2all between both prefill nodes and decode nodes. |
| `sglang-single-node-kimi25.skill` | Single-node SGLang benchmark experiments for Kimi2.5. |
| `sglang-mimo-v2-flash.skill` | Deploys mimo-v2-flash on an existing AWS SageMaker HyperPod cluster. Includes single-node, PD (1P1D), and MTP-enabled deployments. PD uses NIXL libfabric backend for KV transfer, based on SGLang v0.5.6.post2 with custom patches (embedded in the skill's Python scripts). |

### vLLM Deployment Skills

| Skill | Description |
|-------|-------------|
| `deploy-vllm-hyperpod-2p1d.skill` | Deploys DeepSeek-V3 using native vLLM (no Ray, no HyperPod Inference Operator) on AWS SageMaker HyperPod EKS with 3 x ml.p5en.48xlarge (H200). 2P1D topology: Prefill spans 2 nodes with TP16/EP16, Decode on 1 node with TP8/EP8. Uses NIXL LIBFABRIC for KV transfer, UCCL-EP for MoE All2All (Prefill: deepep_high_throughput, Decode: deepep_low_latency). Best config (CUDA Graph + UCCL-EP inflight=16) achieves 404.74 tok/s output throughput — 6x over eager baseline, TPOT reduced from 107ms to 14.6ms (7.3x improvement). |
| `Deepseek-v4-Pro-VLLM-skills.zip` | Deploys DeepSeek-V4-Pro using VLLM on AWS H200 GPU instances (Non-PD and PD disaggregation). |

### EKS Cluster & Model Deployment / Model Training Skills

| Skill | Description |
|-------|-------------|
| `eks-h200-gpu.skill` | Creates an AWS EKS cluster with an H200 GPU node group and runs a simple Kubeflow PyTorch training job. |
| `eks-b300-gpu.skill` | Deploys DeepSeek-V3 on an existing EKS cluster using B300 instances. Includes 1P1D (TP8 EP8, NIXL KV transfer), Non-PD (TP16 EP16), and NCCL allreduce/all2all tests. |
| `eks-b200.skill` | End-to-end EKS + p6-b200.48xlarge cluster setup and DeepSeek-V3 671B FP8 inference deployment. Covers SGLang PD disaggregation (1P1D/2P1D/2P2D/1P2D) with NIXL LIBFABRIC over EFA RDMA, NCCL tests, PyTorchJob distributed training, and 13 known issue troubleshooting guides. |
| `eks-h200-megatron-qwen3-235b-a22b.skill` | Megatron-LM distributed training of Qwen3-235B-A22B (MoE, 128 experts, top-8) on EKS HyperPod H200. Full pipeline: HF model download, megatron-bridge checkpoint conversion, training data preparation, 4-node 32-GPU 5D parallel training (TP/PP/CP/EP/DP). Includes PlanD (TP4/PP1/CP8/EP32, seqlen 64K) and PlanE (TP4/PP2/CP2/EP16/DP2, seqlen 20K), plus UCCL-EP flex vs NCCL alltoall dispatcher performance comparison (UCCL-EP shows +12.7% throughput on PlanE, +7.7% on PlanD; for PlanE EP16 with seqlen 48K, UCCL-EP achieves 20%+ throughput advantage and ~18%+ iteration time improvement over NCCL). |
| `eks-h200-deepseek-v4-pd.skill` | Deploys DeepSeek-V4-Pro using SGLang on AWS H200 instances, covering both Non-PD and PD disaggregation approaches. |

### EC2 & SageMaker Skills

| Skill | Description |
|-------|-------------|
| `ec2-g7e-docker-sglang-2p2d.skill` | Deploys a 4-node 2P2D configuration on AWS G7e.48xlarge EC2 instances using Docker containers. |
| `sagemaker-hyperpod-on-eks-setup.skill` | Automates creation of a SageMaker HyperPod on EKS cluster in AWS Global regions via Claude Code. |

### Presentation

| File | Description |
|------|-------------|
| `GPU Infra As Claude Code Skills.pptx` | Slide deck covering the overall approach and architecture. |


## How to Use

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
2. Place the `.skill` file(s) in your Claude Code skills directory
3. Invoke the skill in Claude Code — it will guide you through the full deployment process including infrastructure provisioning, model downloading, configuration, and benchmarking

## Prerequisites

- AWS account with appropriate GPU instance quotas (H200, B200, B300, G7e depending on the skill)
- AWS CLI configured with proper credentials
- `kubectl`, `eksctl`, and `helm` for EKS-based skills
- Docker for container-based deployments
- Sufficient HuggingFace access for gated model downloads (DeepSeek-V3, Qwen3, etc.)
