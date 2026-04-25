GPU Infra as Claude code skills

* 当前这里提供了几个打包的skill：
   * 1. sglang-deepseek-non-PD.skill ------ 这个skill使用Non-PD部署方式，针对Deepseek-v3 FP8模型，使用单节点部署，2节点部署进行了实验。2节点部署使用了基于NCCL的，以及基于UCCL-EP的方式（使用Non-PD的uccl-ep方式来主要是做一下对比实验）。
   * 2. sglang-deepseek-1p1d.skill ------ 这个skill使用1P1D部署方式，使用NIXL KV transfer engine的两个不同backend即libfabric backend和ucx backend来对比。
   * 3. sglang-2p2d-nccl-nixl.skill ------- 这个skill使用2P2D部署方式，严格说是4台GUP实例，2个P之间通过NCCL通信，2个D之间通过NCCL通信。
   * 4. sglang-2p2d-ucclep-nixl.skill ------- 这个skill使用2P2D部署，两种方式：一种是2个P之间单独独立（不需要通信），2个D之间使用UCCL-EP来做all2all通信；另一种是2个P之间使用UCCL-EP来做all2all通信，2个D之间也使用UCCL-EP来做all2all通信。
   * 5. sglang-single-node-kimi25.skill ------ 这个skill使用单节点对Kimi2.5进行了一些SGLang的benchmark实验。
   * 6. sagemaker-hyperpod-on-eks-setup.skill ---- 这个skill是借助Claude code来在AWS Global region创建Sagemaker hyperpod on EKS集群。
   * 7. ec2-g7e-docker-sglang-2p2d.skill ------ 这个skill是在AWS GPU EC2 G7e.48xlarge 实例上，基于docker container做4节点2P2D部署。
   * 8. eks-h200-gpu.skill ---- 这个skill是创建AWS EKS集群，并创建H200 GPU实例的node group，并跑一个简单的kubeflow pytorch training job。
   * 9. sglang-mimo-v2-flash.skill ---- 这个skill是借助Claude code在已有的AWS sagemaker hyperpod集群上部署mimo-v2-flash，包括单机部署和PD（1P1D）部署以及开启MTP的部署。PD部署使用的NIXL libfabric backend做KV transfer，方案基于SGLang比较旧的版本0.5.6.post2打了很多patch，patch在skill中的python脚本中。
   * 10. eks-b300-gpu.skill ----- 这个skill是在已有的eks集群中使用b300实例来做2节点部署deepseek-v3的，包括1P1D（TP8 EP8，使用NIXL KV transfer）, Non PD（TP16 EP16）的部署方案，以及Nccl-test allreduce和all2all的测试。
   * 11. eks-b200.skill ----- EKS + p6-b200.48xlarge 集群搭建与 DeepSeek-V3 671B FP8 推理部署（SGLang PD disaggregation 1P1D/2P1D/2P2D/1P2D，nixl LIBFABRIC over EFA RDMA），含 NCCL 测试、PyTorchJob 分布式训练及 13 个已知问题排障。


* 小结 for Deepseek-v3（对于当前这个测试场景和已测试过的方案）：
    * 用单机部署性能最好，性价比最高。
        *  1-node EP8 TP8 — 仅 1 台机器就能达到 687 tok/s，TPOT 12ms。
        * 注意：那是不是针对这个测试workload，就一定是单机部署性价比最好？
            * 不一定。因为我们并没有穷尽的去测试XPXD的情况。
    * 对于2台节点部署的方案，综合性能最好的是 1P1D + NIXL Libfabric backend + NCCL。
    * 对于4节点部署的方案，2P2D + UCCL-EP的方式要好于2P2D + NCCL的方式。
        * UCCL-EP/DeepEP都是和PD分离配合起来才能更好的发挥作用。
    * 对于4节点部署的2P2D方案，方案（2P是单独的，2D之间使用UCCL-EP low latency mode ）比方案（2P之间使用UCCL-EP normal mode，2D之间使用UCCL-EP low latency mode）更好。
        * 这个也和蚂蚁金服针对deepseek-v3使用4台H20建议的部署方案是一致的结论。
            * 独立 prefill（TP=8 单节点）+ flashmla decode 是 2P2D 的最优架构，避免 prefill 跨节点通信开销，并利用 flashmla 优化 MLA decode。
        * 2P2D，P之间单独部署，D之间使用UCCL-EP，参数配置基本和蚂蚁(https://github.com/antgroup/sglang/pull/4)的一个建议类似：
            * 区别：disable prefix cache，没有使用speculative相关参数，使用的是H200 GPU，使用的EFA和UCCL-EP。

