# BioContainers

Cluster nodes automount a CernVM-File System at
`/cvmfs/singularity.galaxyproject.org/`. This provides a universal file system
namespace to Singularity images for the BioContainers project, which comprises
container images automatically generated from Bioconda software packages. The
Singularity images are organized into a directory hierarchy following the
convention:

```bash
/cvmfs/singularity.galaxyproject.org/FIRST_LETTER/SECOND_LETTER/PACKAGE_NAME:VERSION--CONDA_BUILD
```

For example:

```bash
singularity exec /cvmfs/singularity.galaxyproject.org/s/a/samtools:1.13--h8c37831_0 samtools --help
```

The Bioconda package index lists all software available in
`/cvmfs/singularity.galaxyproject.org/`, while the BioContainers registry provides
a searchable interface.

**NOTE**: There will be a 10-30 second delay when first accessing
`/cvmfs/singularity.galaxyproject.org/` on a compute node on which it is not
currently mounted; in addition, there will be a delay when accessing a
Singularity image on a compute node where it has not already been accessed and
cached to node-local storage.


## BioContainers images in Docker Hub

A small number of Biocontainers images are available only in [DockerHub
](https://hub.docker.com/u/biocontainers) under the
biocontainers organization, and are **not** available on Cannon under
`/cvmfs/singularity.galaxyproject.org/`. 

See [BioContainers GitHub](https://github.com/BioContainers/containers) for a
complete list of BioContainers images available in DockerHub (note that many of
the applications listed in that GitHub repository have since been ported to
Bioconda, but a subset are still only available in DockerHub).

These images can
be fetched and built on Cannon using the `singularity pull` command:

```bash
singularity docker://biocontainers/<image>:<tag>
```

For example, for the container `cellpose` with tag `2.1.1_vc1` ([cellpose
Docker Hub page](https://hub.docker.com/r/biocontainers/cellpose/tags)):

```bash
singularity pull --disable-cache docker://biocontainers/cellpose:2.1.1_cv1
[jharvard@holy2c02302 bio]$ singularity pull --disable-cache docker://biocontainers/cellpose:2.1.1_cv1
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
2023/03/13 15:58:16  info unpack layer: sha256:a603fa5e3b4127f210503aaa6189abf6286ee5a73deeaab460f8f33ebc6b64e2
2023/03/13 15:58:17  info unpack layer: sha256:b00aaacf759c581712fa578a6b4e8e0b9fc780919a5d835a168457b754755644
2023/03/13 15:58:17  info unpack layer: sha256:372d780866a7e569457582348fbf850edc018b6b015335a4a56403fe299ff04b
2023/03/13 15:58:17  info unpack layer: sha256:feb836cf9ff261a0d9feb57f8808540cbb140d6d9e957af5daad2767d65fec36
2023/03/13 15:58:17  info unpack layer: sha256:6a0e4abca74a97205cba7ecb141fd4210bfab67dddb55e3a1fdaa6bcefbc44de
2023/03/13 15:58:17  info unpack layer: sha256:74be8ddfa7b55475808017bc4d978a439c2eeaeb252027c39bd6f29259355993
2023/03/13 15:58:20  info unpack layer: sha256:0a112f1bbdd8edefa7d2a40ad647c75067bb501ab4a3bfdacb5166727382db74
2023/03/13 15:58:20  info unpack layer: sha256:4012abce08aacd374ad001e1f9fd6d9fe864b01eadcf978306c70b4407b5c17b
2023/03/13 15:58:20  info unpack layer: sha256:2e0aab8a3f9f8d9f50931dfb7e8f751e0909249e82f56632a9bb6d2509387f74
2023/03/13 15:58:22  info unpack layer: sha256:d0819e2d86c5e671097843a5448232b3eff97d2685ab1ef97feac45be4ae54cc
2023/03/13 15:58:23  info unpack layer: sha256:376d1cd1a59141b316dc49fae4153aeaf5aa47f7e174651ca8bca98e734bd5f2
2023/03/13 15:58:59  info unpack layer: sha256:7bef5b9ba6503c93f4ad31d5664e7cffa42dd2b5368158a022d543e07979115b
2023/03/13 15:59:00  info unpack layer: sha256:75dc70b21417624a19747e6f6706571348f7c388a6efbb8a646da3e13b95a6de
2023/03/13 15:59:01  info unpack layer: sha256:bad0c0ea6165e4bb565161ee2d7779fd0555657ecf494f50c9a9710acfefcbe3
INFO:    Creating SIF file...
```

The `sif` image file `cellpose_2.1.1_cv1.sif` will be created:

```bash
[jharvard@holy2c02302 bio]$ ls -lh
total 2.5G
-rwxr-xr-x 1 jharvard jharvard_lab 2.4G Mar 13 15:59 cellpose_2.1.1_cv1.sif
-rwxr-xr-x 1 jharvard jharvard_lab  72M Mar 13 12:06 lolcow_latest.sif
```

## BioContainer and package tips

- The registry https://biocontainers.pro may be slow
- We recommend to first check the [Bioconda package
  index](https://bioconda.github.io/conda-package_index.html), as it quickly
  provides a complete list of Bioconda packages, all of which have a
  corresponding  biocontainers image in `/cvmfs/singularity.galaxyproject.org/` 
- If an image doesn't exist there, then there is a small chance
  there might be one generated from a Dockerfile in [BioContainer GitHub](https://github.com/BioContainers/containers)
- If your package is listed in the BioContainer GitHub, search for the package
  in Docker Hub, under the [biocontainers
  organization](https://hub.docker.com/u/biocontainers)(e.g. search for
  `biocontainers/<package>`)  
