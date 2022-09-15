## Installing Software Yourself

#### *NOTE: This documentation assumes you are familiar with the module system. See [this page](https://docs.rc.fas.harvard.edu/kb/modules-intro/) for details.*

<h3><b><a name="contents">Table of Contents:</a></b></h3>
<ul>
<li><a href="#intro" style="text-decoration:none"><b>Introduction</b></a></li>

<li><a href="#gnu" style="text-decoration:none"><b>Installing GNU-toolchain-style apps</b></a></li>
<li><a href="#cmake" style="text-decoration:none"><b>Cmake</b></a></li>
<li><a href="#python" style="text-decoration:none"><b>Python packages</font></b></a></li>
<li><a href="#r" style="text-decoration:none"><b>R packages</b></a></li>
<li><a href="#containers" style="text-decoration:none"><b>Singularity & Docker containers</b></a></li>
</ul>

<h3><b><a name="intro">Introduction</a></b></h3>

We have hundreds of high performance, scientific computing applications installed and available through our [module system](https://docs.rc.fas.harvard.edu/kb/modules-intro/). If there’s a software app that you’d like to use, is commonly used by a group of labs or your scientific domain, and is not available on the cluster, we encourage you to request that we install it as a module by submitting a [help request on the RC Portal](https://portal.rc.fas.harvard.edu/rcrt/submit_ticket). We want the cluster to be as immediately usable as possible for all users, and having all the common applications available is part of that.
If you are looking to use a "bleeding edge" version of software or something specific only to your lab, you can install applications yourself in your home directory, any other space to which you can write, or a shared lab directory. Writing and installing your own software may be part of your research, too. Below you will find several approaches to compiling and installing personal software on the cluster. If you are having problems during your installation, submit a [help request on the RC Portal](https://portal.rc.fas.harvard.edu/rcrt/submit_ticket) or stop by [Office Hours](https://www.rc.fas.harvard.edu/training/office-hours/).

<div align="right"><a href="#contents" style="text-decoration:none"><font color ="blue">Back to top</font></a></div>

<h3><b><a name="gnu">Installing GNU-toolchain-style apps</a></b></h3>

Many scientific applications are provided as a source code and require conversion to binary form — a process called <code>compilation</code> or <code>building</code>. Such applications are often packaged to work with the [GNU toolchain](https://en.wikipedia.org/wiki/GNU_toolchain) — you use the commands <code>./configure</code>, <code>make</code>, and <code>make install</code> to compile and install them. When installing software it is important to specify an installation location, which is writable by you. This is done by the <code>--prefix</code> option specified on the <code>./configure</code> command-line. The below example illustrates the installation of software in the directory <code>$HOME/sw</code>.

* Software source is usually distributed in a compressed "tarball" such as a <code>myapp-x.y.z.tar.gz</code> or <code>myapp-x.y.z.tar.bz2</code> file. The first step is to download the software source (a "tarball").

* The next step is to unpack the "tarball" Use <code>tar zxvf myapp-x.y.z.tar.gz</code> for gzipped tarball or <code>tar jxvf myapp-x.y.z.tar.bz2</code> for a bzipped tarball.

* Next cd in the unpacked directory:
<pre>cd myapp-x.y.z</pre>

**Note:** Often the newly created directory won’t exactly match the tarball, or the files will have been dumped to the current working directory; adjust accordingly.

* Configure the software, including telling it to install to somewhere in your home directory, e.g.,
<pre>./configure --prefix=$HOME/sw</pre>

**Note:** If you want to install every app in its own directory, instead of having them all share one (e.g. if you want to have different versions of the same software available), use a more specific prefix, like <code>--prefix=$HOME/sw/myapp-x.y.z</code>.
Some apps don’t actually use the full GNU toolchain, and don’t have a configure script -- they have just a <code>Makefile</code>. In that case, you’ll have to look at the <code>Makefile</code> to see if it has an adjustable installation location. If not, you’ll have to manually install files.

<div align="right"><a href="#contents" style="text-decoration:none"><font color ="blue">Back to top</font></a></div>