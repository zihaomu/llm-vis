# HardwareProfile 示例

[`synthetic-bf16.json`](synthetic-bf16.json) 是完全虚构的自含测试数据，不对应任何 AMD、NVIDIA 或其他真实设备，也不是 LLM-Vis 默认硬件。

Roofline 调用方必须显式提供：

- dtype 对应的 peak FLOPs/s；
- memory bandwidth bytes/s；
- profile dtype；
- provenance（vendor spec、measured、user supplied 或 synthetic）。

缺失峰值保持 Unknown。输出只有 compute lower bound、bandwidth lower bound、两者的 max lower bound、arithmetic intensity 和 bottleneck class；它们不是 latency estimate，不包含 Kernel launch、cache、fusion、occupancy、同步、调度或真实 HBM traffic。

