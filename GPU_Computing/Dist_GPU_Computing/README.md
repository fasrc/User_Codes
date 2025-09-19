# Distributed GPU Computing

The FAS cluster *Cannon* supports distributed GPU workloads on both single and multiple nodes, making it well-suited for deep learning, large language model (LLM) training, and large-scale GPU-accelerated simulations. On a single node, jobs can scale efficiently across multiple GPUs using MPI+CUDA or NCCL+CUDA, providing high-throughput parallelism and optimized GPU-to-GPU communication. For multi-node execution, the same frameworks extend naturally over Cannon’s high-speed interconnect, allowing seamless scaling from a few GPUs to dozens of nodes.

Common workflows include multi-GPU training with MPI and NCCL-based collectives for intra- and inter-node communication. Large-scale LLM training typically combines NCCL with MPI for process orchestration, ensuring efficient gradient synchronization across GPUs. Example job scripts in this repository show how to request GPUs with Slurm, configure CUDA/NCCL/MPI environments, and launch distributed jobs for both single-node and multi-node runs, serving as templates for adapting to specific workloads.

## Examples

* [Using MPI and CUDA](./MPI_and_CUDA/)
* [Using NCCL and CUDA on a single node](./NCCL_and_CUDA/SingleNode)
* [Compiling and running KHARMA (MPI+CUDA code)](https://github.com/fasrc/User_Codes/tree/master/Applications/KHARMA)