<img src="Images/mamba-logo.png" alt="mamba-logo" width="250"/>

[Mamba](https://mamba.readthedocs.io/en/latest/index.html#) is a fast, robust, and cross-platform package manager.

<code>mamba</code> is a drop-in replacement and uses the same commands and configuration options as <code>conda</code>. You can swap almost all commands between conda & mamba.

<code>Mamba</code> is available on the FASRC cluster as a software module:

```bash
$ module load Mamba/4.14.0-0
$ python -V
Python 3.10.6
```

You can create conda environments with mamba in the same way as with conda:

```bash
$ mamba create -n ENV_NAME PACKAGES
```

> **NOTE:** The major advantage of using mamba instead of conda is that the environment creation, and installing / uninstalling of packages is *much* faster with mamba, e.g.,

```bash
$ mamba create -n python_env1 python=3.10 pip wheel

                  __    __    __    __
                 /  \  /  \  /  \  /  \
                /    \/    \/    \/    \
███████████████/  /██/  /██/  /██/  /████████████████████████
              /  / \   / \   / \   / \  \____
             /  /   \_/   \_/   \_/   \    o \__,
            / _/                       \_____/  `
            |/
        ███╗   ███╗ █████╗ ███╗   ███╗██████╗  █████╗
        ████╗ ████║██╔══██╗████╗ ████║██╔══██╗██╔══██╗
        ██╔████╔██║███████║██╔████╔██║██████╔╝███████║
        ██║╚██╔╝██║██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║
        ██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║
        ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝

        mamba (0.25.0) supported by @QuantStack

        GitHub:  https://github.com/mamba-org/mamba
        Twitter: https://twitter.com/QuantStack

█████████████████████████████████████████████████████████████


Looking for: ['python=3.10', 'pip', 'wheel']

pkgs/main/noarch                                              No change
pkgs/r/linux-64                                               No change
pkgs/r/noarch                                                 No change
pkgs/main/linux-64                                   5.3MB @   4.5MB/s  1.2s
conda-forge/noarch                                  11.5MB @   3.8MB/s  3.1s
conda-forge/linux-64                                30.1MB @   4.4MB/s  7.1s
Transaction

  Prefix: /scratch/pkrastev/conda/python_env1

  Updating specs:

   - python=3.10
   - pip
   - wheel


  Package                Version  Build            Channel                  Size
──────────────────────────────────────────────────────────────────────────────────
  Install:
──────────────────────────────────────────────────────────────────────────────────

  + _libgcc_mutex            0.1  main             pkgs/main/linux-64     Cached
  + _openmp_mutex            5.1  1_gnu            pkgs/main/linux-64     Cached
  + bzip2                  1.0.8  h7b6447c_0       pkgs/main/linux-64     Cached
  + ca-certificates   2023.01.10  h06a4308_0       pkgs/main/linux-64     Cached
  + certifi            2022.12.7  py310h06a4308_0  pkgs/main/linux-64     Cached
  + ld_impl_linux-64        2.38  h1181459_1       pkgs/main/linux-64     Cached
  + libffi                 3.4.2  h6a678d5_6       pkgs/main/linux-64     Cached
  + libgcc-ng             11.2.0  h1234567_1       pkgs/main/linux-64     Cached
  + libgomp               11.2.0  h1234567_1       pkgs/main/linux-64     Cached
  + libstdcxx-ng          11.2.0  h1234567_1       pkgs/main/linux-64     Cached
  + libuuid               1.41.5  h5eee18b_0       pkgs/main/linux-64     Cached
  + ncurses                  6.4  h6a678d5_0       pkgs/main/linux-64     Cached
  + openssl               1.1.1t  h7f8727e_0       pkgs/main/linux-64     Cached
  + pip                   23.0.1  py310h06a4308_0  pkgs/main/linux-64        3MB
  + python                3.10.9  h7a1cb2a_2       pkgs/main/linux-64       28MB
  + readline                 8.2  h5eee18b_0       pkgs/main/linux-64     Cached
  + setuptools            65.6.3  py310h06a4308_0  pkgs/main/linux-64        1MB
  + sqlite                3.40.1  h5082296_0       pkgs/main/linux-64     Cached
  + tk                    8.6.12  h1ccaba5_0       pkgs/main/linux-64     Cached
  + tzdata                 2022g  h04d1e81_0       pkgs/main/noarch       Cached
  + wheel                 0.38.4  py310h06a4308_0  pkgs/main/linux-64       66kB
  + xz                    5.2.10  h5eee18b_1       pkgs/main/linux-64     Cached
  + zlib                  1.2.13  h5eee18b_0       pkgs/main/linux-64     Cached

  Summary:

  Install: 23 packages

  Total download: 32MB

──────────────────────────────────────────────────────────────────────────────────

Confirm changes: [Y/n] Y
wheel                                               65.6kB @ 337.5kB/s  0.2s
setuptools                                           1.2MB @   5.4MB/s  0.2s
pip                                                  2.7MB @  10.7MB/s  0.3s
python                                              28.1MB @  73.0MB/s  0.4s
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

To activate this environment, use

     $ mamba activate python_env1

To deactivate an active environment, use

     $ mamba deactivate
```

To use the environment, do:

```bash
$ source activate python_env1
```

You can list the packages currently installed in the conda environment with:

```bash
$ mamba list
```

You can install new packages with:

```bash
$ mamba install -y numpy
```

To uninstall packages, use:

```bash
$ mamba uninstall PACKAGE
```

When you finish using the conda environment, you can deactivate it with:

```bash
$ conda deactivate
```

For additional features, please refer to the [Mamba documentation](https://mamba.readthedocs.io/en/latest/index.html).

## Conda environments in Lab space


