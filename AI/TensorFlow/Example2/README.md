### Purpose:
An example to start with [TesorBoard](https://www.tensorflow.org/tensorboard) (TB) on the FASRC Cannon cluster. The specific example is addapted from [here](https://www.tensorflow.org/tensorboard/get_started).

**NOTE 1:** Due to a <code>GLIBC 2.18</code> dependency issue with Centos 7, <code>TensorBoard</code> currently works *only* in Jupyter notebooks.

**NOTE 2:** TB needs to be launched with, e.g, <code>%tensorboard --logdir logs/fit --bind_all</code> (add the option <code>--bind_all</code>).

### Contents:

*   <code>get_started.ipynb</code>: Jupyter notebook