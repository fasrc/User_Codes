# Augustus installation using Spack

* Go to a compute node. For example:
```bash
salloc -p shared --mem-per-cpu 4g -t 0-03:00 -c 4
```

* Install Spack if already not installed. See our [Spack
  instructions](https://docs.rc.fas.harvard.edu/kb/spack/). For
  example, to install it in your `$HOME`:
```bash
    cd ~
    git clone -c feature.manyFiles=true https://github.com/spack/spack.git
```    

* Activate Spack:
```bash
. share/spack/setup-env.sh
```
    One can put this command in their `~/.bashrc` to ensure that Spack is activated every time one logs in.  

* Install Augustus:
```bash
spack install augustus
```

Upon successful installation, one should see:
```bash
==> augustus: Successfully installed augustus-3.5.0-ia3gaqksuzg5zbxlbsawrkybipxn6s6h

  Stage: 15.84s.  Edit: 0.25s.  Build: 1m 27.66s.  Install: 7m 6.07s.  Post-install: 41.17s.  Total: 9m 43.68s

[+] ~/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/augustus-3.5.0-ia3gaqksuzg5zbxlbsawrkybipxn6s6h
```

* Test installation. For example: 
```bash
spack load augustus
augustus --help
```
