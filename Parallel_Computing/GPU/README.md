# GPU Computing on FAS-RC Cannon Cluster 

<img src="Images/GPU-logo.jpeg" alt="GPU-logo" width="200"/>

GPU computing refers to the use of graphics processing units (GPUs) for computational tasks beyond their original purpose of rendering graphics in video games and other visual applications. GPUs have thousands of processing cores that can perform many simple calculations simultaneously, making them particularly well-suited for tasks that can be broken down into many smaller operations that can be executed in parallel. This parallelism allows for much faster processing times than traditional central processing units (CPUs), which are designed for more general-purpose computing.

One of the main advantages of GPU computing is that it can significantly reduce the time required to perform computationally intensive tasks, such as simulating physical systems, training machine learning models, and processing large amounts of data. This has enabled many new applications in fields such as scientific research, finance, and healthcare. For example, GPUs have been used to simulate the behavior of complex materials, to analyze financial market data, and to speed up medical image processing.

The FAS-RC Cannon cluster provides access to a variety of NVIDIA GPUs, including Tesla A100 and V100 models. Researchers can submit jobs to the cluster and take advantage of the GPUs to accelerate their computations, allowing them to perform simulations, run deep learning models, and process large datasets more quickly and efficiently than on traditional CPUs. Additionally, the Cannon cluster provides a software stack that includes popular GPU programming frameworks such as CUDA and cuDNN, making it easier for researchers to develop and run GPU-accelerated code.

## Examples

* [Using GPU enabled numerical libraries](./Libs)
* [OpenACC](./OpenACC)
* [Using Programming Languages](./CUDA)
