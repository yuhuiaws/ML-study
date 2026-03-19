GPU Infra as Claude code skills

当前这里提供了几个打包的skill：
1. sglang-deepseek-non-PD.skill ------ 这个skill使用Non-PD部署方式，针对Deepseek-v3 FP8模型，使用单节点部署，2节点部署进行了实验。2节点部署使用了基于NCCL的，以及基于UCCL-EP的方式（使用Non-PD的uccl-ep方式来主要是做一下对比实验）。
2. sglang-deepseek-1p1d.skill ------ 这个skill使用1P1D部署方式，使用NIXL KV transfer engine的两个不同backend即libfabric backend和ucx backend来对比。
3. sglang-2p2d-nccl-nixl.skill ------- 这个skill使用2P2D部署方式，严格说是4台GUP实例，2个P之间通过NCCL通信，2个D之间通过NCCL通信。
4. sglang-2p2d-ucclep-nixl.skill ------- 这个skill使用2P2D部署，两种方式：一种是2个P之间单独独立（不需要通信），2个D之间使用UCCL-EP来做all2all通信；另一种是2个P之间使用UCCL-EP来做all2all通信，2个D之间也使用UCCL-EP来做all2all通信。
5. sglang-single-node-kimi25.skill ------ 这个skill使用单节点对Kimi2.5进行了一些SGLang的benchmark实验。
6. sagemaker-hyperpod-on-eks-setup.skill ---- 这个skill是借助Claude code来在AWS Global region创建Sagemaker hyperpod on EKS集群。
