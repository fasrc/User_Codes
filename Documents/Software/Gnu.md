## Installing GNU-toolchain-style apps
Many apps are provided in source code form and require conversion to binary form — a process called compilation or building. Such apps are often packaged to work with the GNU toolchain — you use the commands <code>./configure</code>, <code>make</code>, and <code>make install</code> to build and install them. The important thing is that you specify a --prefix in order to write the software to a non-default location, i.e. a location writable by you. The examples below install software to a directory <code>/n/holylabs/LABS/jharvard_lab/LAB/software/</code>. Before looking at the instructions below consult the installation instructions for the software you are looking at as they will have more details about code features.

Software source is often distributed as a zipped tarball, such as <code>APP-X.Y.Z.tar.gz</code>. The first step is to unpack it. This can be done in whatever directory you like, even a temporary space that you later delete, since the installation step below will copy stuff out of this particular directory and into the prefix you specify:

```bash
tar -zxvf APP-X.Y.Z.tar.gz
```

Next, <code>cd</code> into the unpacked directory:

```bash
cd APP-X.Y.Z
```

Often the newly created directory won’t exactly match the tarball, or the files will have been dumped to the current working directory; adjust accordingly.

Next, configure the software. You will want to run <code>./configure --help</code> to see if there are any addition options you want to enable. An option you will want to set is the <code>--prefix</code> flag telling it to install to somewhere in your lab directory:

```bash
./configure --prefix=/n/holylabs/LABS/jharvard_lab/LAB/software/
```

If you want to install every app in its own directory, instead of having them all share one (e.g. if you want to have different versions of the same software available), use a more specific prefix, like <code>--prefix=/n/holylabs/LABS/jharvard_lab/LAB/software/APP-X.Y.Z</code>

Some apps don’t actually use the full GNU toolchain, and don’t have a configure script; they have just a <code>Makefile</code>. In that case, you’ll have to look at the Makefile to see if it has an adjustable installation location. If not, you’ll have to manually install files.

Next, build the software:

```bash
make
```

This is the step that actually compiles the software. Once you work through any issues that may come up and the software compiles successfully, install it:

```bash
make install
```

If you get any Permission denied issues at this point, it’s possible the software did not properly use the <code>--prefix</code> you have it. Likewise, you may get the error <code>No rule to make target `install'</code>. In fact, many packages only mimic the GNU-toolchain style but actually work slightly differently. In such cases, you’ll have to manually copy the build outputs to their destination.