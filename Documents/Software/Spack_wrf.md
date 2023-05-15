# Build WRF using Spack

<img src="Images/spack-logo.svg" alt="spack-logo" width="100"/>

The Weather Research and Forecasting (WRF) Model is a next-generation mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications. WRF features two dynamical cores, a data assimilation system, and a software architecture supporting parallel computation and system extensibility. The model serves a wide range of meteorological applications across scales from tens of meters to thousands of kilometers.

WRF official website: https://www.mmm.ucar.edu/weather-research-and-forecasting-model

## Compiler and MPI Library Spack configuration

We use the [Intel compiler suite together with the Intel MPI Library](https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Spack_Intel-MPI.md). The below instructions assume that spack is already configured to use the Intel compiler `intel@2021.8.0` and Intel MPI Library `intel-oneapi-mpi@2021.8.0`.

## Create WRF spack environment and activate it

```bash
spack env create wrf
spack env activate -p wrf
```

## Add the required packages to the spack environment

In addition to `WRF` and `WPS` we also build `ncview` and `ncl`.

```bash
spack add intel-oneapi-mpi@2021.8.0
spack add hdf5@1.12%intel@2021.8.0 +cxx+fortran+hl+threadsafe
spack add libpng@1.6.37%intel@2021.8.0
spack add jasper@1.900.1%intel@2021.8.0
spack add netcdf-c@4.9.0%intel@2021.8.0
spack add netcdf-fortran@4.6.0%intel@2021.8.0 
spack add xz@5.4.2%intel@2021.8.0
spack add wrf@4.4%intel@2021.8.0
spack add wps@4.3.1%intel@2021.8.0
spack add cairo@1.16.0%gcc@8.5.0
spack add ncview@2.1.8%intel@2021.8.0
spack add ncl@6.6.2%intel@2021.8.0
```
> **NOTE:** Here we use the `gcc@8.5.0` compiler to build `cairo@1.16` as it fails to compile with the Intel compiler.

## Install the WRF environment

Once all required packages are added to the environment, it can be installed with:

```bash
spack build
```
For example,

```bash
[wrf] [pkrastev@builds01 spack]$ spack install
==> Installing environment wrf
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libiconv-1.16-iidmiswunjmyfgqv5im7dbq3vxmfugvn
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libsigsegv-2.13-62qs56bhpwobu346x7njysywttoj6yf4
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/berkeley-db-18.1.40-si74jisjm3z5hktwugiuvmbrlyd54ziy
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libiconv-1.16-tctvch3rmfupkx3bq3nbreut6vbes3av
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pkgconf-1.8.0-lcbimx5pnerubwp2avwnqjssnas3r63j
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/zlib-1.2.13-eblkljcxuaf72maibqyrsnj3bqn3lo2b
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pkgconf-1.8.0-p4zr5nbn2qfnm2ezj3yuna7z7ldxugfo
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xz-5.4.2-wnecuvf4a4shzeaymtpfjfvjgb3vjcyx
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/zstd-1.5.2-nsa6ewz4bkbbuayimbxjzml2iif2vgj5
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libffi-3.4.2-nfk6v4bok5jjbpgynuwibwklq2eni5k4
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libmd-1.0.4-nsvbozgifetn6onk2rgb6yny3ycv2scg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ca-certificates-mozilla-2022-10-11-rtn42qshvxwqcinxa7fetl6dej4polau
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pcre2-10.39-phldzodlzipj6jf6odvske5zx464okjc
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/util-macros-1.19.3-i44qvmbd76rv3yer4cteg2l5kwnjiit4
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libpthread-stubs-0.4-kpfdvil6itiz4urytgi2humcjewzmylw
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xcb-proto-1.14.1-zmtlojpj26autleu22zvy4jxdxqxv3dm
[+] /n/sw/intel-oneapi-2023 (external intel-oneapi-mpi-2021.8.0-xfvjrn2fuyum7xtnyz7g3gjv3c5d27tf)
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ca-certificates-mozilla-2022-10-11-rlli5fmtpjaatq4lseetdudmhhcilc43
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/berkeley-db-18.1.40-m6r6onkubysvl3gxtw7t7tpghoq6x6g6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/zlib-1.2.13-admgboe6z4bq56lnw3owqrwqydoh2qkv
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/findutils-4.9.0-74hvmuixshzcogzywohjwb6xaybw5weu
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/gperf-3.1-evwzpy2iofahxg6b7nigvqho4cgels5s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/nasm-2.15.05-7yzvgoewglxmapdfgld4k4thgwronxb2
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/diffutils-3.8-idtn6tigrq7qhdqlofjawd5jax6ugen6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/diffutils-3.8-6evfnay4rp5awqlffw3rzrje5kvvssqw
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ncurses-6.3-wqhrq3qaafs3k2d6ohmvd25ezkmprwdz
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/util-linux-uuid-2.38.1-domgwguxjj3x3zxj6g6myrwy3gho5xts
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libpng-1.6.37-ye47tftsgy4tvla7z3oc4c6bgwhc4xey
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pigz-2.7-l6a26fe5mzh4jayaovjcjilfum6xojcn
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ncurses-6.3-wq3qyyg5nbr3r4rbpxtm2w6sz2asqwmf
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxml2-2.10.1-o2vey6cv3ugjdtspjlcskf4rx5ws7s24
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libbsd-0.11.5-felxkikh7nhslshyq7orqzssf6dcdrqv
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/kbproto-1.0.7-j33brt53vnaw3zaotxabxb5go6pdyc7c
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xtrans-1.3.5-sqwgeql64i72tvwmou3dlbh3jpppt25w
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/renderproto-0.11.1-hoyhdkkadvndnhbyebyetsawvta5zkdg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/inputproto-2.3.2-qgktpjheu46454gytnxcjvfyssuk6ols
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xproto-7.0.31-tegjorqu7sisv76l4fpangwumssqtse6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/fontsproto-2.1.3-mm5urtoshpzqscnaeb5rjoz6fwtudzkj
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xextproto-7.3.0-ny4gtxdrwutoq25zxp5e2mo6bavzma4g
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/m4-1.4.19-74oyqu4jrjwaw3okp6dfslbcx6qeyzdh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/bzip2-1.0.8-q4kqkohouj2iwivc2kav2imqmkkime3k
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/bzip2-1.0.8-gqfgt2fjlfethml6ajjhxarvghhg57b4
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/readline-8.1.2-c5p66oj5jesewxxkk7bjdfds66wnredk
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/tcsh-6.24.00-4fuje5hscpcskh3sluzcgcixprajjeof
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pixman-0.40.0-ve2vgqgygr5gqsbh7vsnb52voqb7xu2h
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/readline-8.1.2-lu2zjeb2y7y7k6trks5bkgswzc5mhz6h
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/expat-2.4.8-5ujmcgjf7vfll22ee5a3rzoilpwpvhkz
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libfontenc-1.1.3-tpk6zkilfo6v3r73hfjgfktepribiaak
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libice-1.0.9-rl6f3sofdbckrxffbaoafc6ytdddhduh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxdmcp-1.1.2-hxqqa7w3u6ip22dvzo35xyj5aty4psc5
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxau-1.0.8-6jdrz5k7rkumyyfij6vyqu4qkyxpbvbe
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/makedepend-1.0.5-oju3cek5whnepbfkzw4dfgcyb7vlpt32
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libtool-2.4.7-ultzx7g4pwfunpuzdnnges3lrxecqryu
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/tar-1.34-lyuz7us46teojd3w5663kwpuspk5qkn4
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/freetype-2.11.1-2s7lipkvdcef346wcq3etphg6zjceero
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/gdbm-1.23-fgiyuwt6k56ogkxzztbdeeqb5mmjgsx3
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/sqlite-3.39.4-elnswenhmv425spmg4fi4dwd5snd7fyb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/gdbm-1.23-h5r23zyhy5plttfjopj6hfn5nxg3526q
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/udunits-2.2.28-tzfqkl75vh5ayefsp6n7jdufiqss7djr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libsm-1.2.3-h7edt44nbf6z6mqtrwv7662te4wszoic
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxcb-1.14-kfrac4vdruypz2zdn5w4w4ykb4gty22x
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/gettext-0.21.1-cb62bb3nl7edr4w5hqas2qqmpouxjbpz
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxfont-1.5.2-gpgpsvajykkl7qhszd5vcjnlzda5lmcc
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/mkfontscale-1.1.2-wzjb5qilbiiyq5qbrducfi2dz3f4tsa7
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/perl-5.36.0-6phwfmc6deklm3ofxriq4y764p6w444l
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/perl-5.36.0-moqx3wcrd3zgpgm4q27phfuxoymdhnkg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/bdftopcf-1.0.5-fbqjd6kluhnnsbyjjilofqkqoxzudmnn
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/mkfontdir-1.0.7-5tfd765wik45qqrz7zj36e2aolwx3bxb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/autoconf-2.69-ge73sgi2pdnb4r5rivr4qk2hkvltmnvz
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/parallel-netcdf-1.12.3-35hdsbsajtdhy6enoycne2sw7iitshci
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libx11-1.7.0-vskloucsqpoidexjdxyhnka6l2to6mlh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/openssl-1.1.1s-w5pspf4xijmqjkymevxkawybicj7ut6j
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/bison-3.8.2-vz7gn2yusekga2vzhu2x47ldnp4uddcc
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/openssl-1.1.1s-rjscbgskagds5vtgqfis7ynma42o2zn5
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/automake-1.16.5-lljzv5n25hk4vrv3ktfcfoj3cxj3ka7j
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxpm-3.5.12-nosstb5xjh6jcpcrtjlgjvvnr3s6pscy
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxrender-0.9.10-myy43d5pnzk2hjsfnxszx6jp3vfctvpk
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxext-1.3.3-s2dmmy2d564dmrwgcwvjtbyjvupunciz
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxt-1.1.5-2jysck5zzdzo4tqzqc6kmhn434gimvg3
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/python-3.10.8-vjuzxhnl7gnakfdbzt2txigb5ngfjln6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/cmake-3.24.3-c2fjg73qx346h72p5gvvsf5zqcycuxpj
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/curl-7.85.0-muxngtscxyzd5rgnl2okfdb3u44w63fw
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/flex-2.6.3-2q55nay6s355iug3zzpiiafs6fwscidn
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/cmake-3.24.3-zj2opbmjgzlp6x3ww5t2m2s6a5fko6sh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/font-util-1.3.2-x754ddudcch7tzxjbmaqih7dyn4m4obm
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxmu-1.1.2-v466dcxqt6gdlerm5mwscombfglwa4mb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/py-pip-22.2.2-lcri4b47lsfxhzfbfl5dnnjxagh7zalm
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ninja-1.11.1-6m7uacx64ufhm7uwhzeoei4tnq4tjodh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/xerces-c-3.2.3-ot4wqn3ymhp3dfg5jswiheee2462kxbg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libaec-1.0.6-iqp56r7ddkbwdcgjhcyxmfif2premib7
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libjpeg-turbo-2.1.3-h6pnkd4hvjp3st2mf2yiumqhymbtsyxw
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/fontconfig-2.13.94-tr34ptuwjgr7iaurcvzznv2r7d34vgxd
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libxaw-1.0.13-w2hfanerrxfknhbq5ttjio6cq4oshvmc
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/py-setuptools-65.5.0-q4pv5jk6xcx5ojbqbn5lhakvsiffbjzs
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/py-wheel-0.37.1-rixpkhfry2h3vo7gschn776hxedt6kkh
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/hdf5-1.12.2-6ma2cfdyygzfnlze65buhgpijns4k2a6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/meson-0.63.3-2huhrbnaq7hkwvur3v2usxg6z462qen7
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/netcdf-c-4.9.0-6guc4ig2wrrblkvffebykdctb5c7eeyy
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/glib-2.74.1-f4cotpiyekxsoab6w76plvbhh3c5orfx
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/netcdf-fortran-4.6.0-fsjz6ttpmlewa6dkisejm55wmzdn5k4m
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/cairo-1.16.0-zlofwcnsgoupzjbp6v2csjagprllkuvr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/esmf-8.3.1-zktq3axpong6gvpin75c3ue6egfv3gg7
==> Installing ncl-6.6.2-b44qiftbaxe2uwrgav54qdwzpgw45dtl
==> No binary for ncl-6.6.2-b44qiftbaxe2uwrgav54qdwzpgw45dtl found: installing from source
==> Using cached archive: /builds/pkrastev/Spack/spack/var/spack/cache/_source-cache/archive/ca/cad4ee47fbb744269146e64298f9efa206bc03e7b86671e9729d8986bb4bc30e.tar.gz
==> Using cached archive: /builds/pkrastev/Spack/spack/var/spack/cache/_source-cache/archive/17/1766327add038495fa3499e9b7cc642179229750f7201b94f8e1b7bee76f8480.zip
==> Moving resource stage
	source: /tmp/pkrastev/spack-stage/resource-triangle-b44qiftbaxe2uwrgav54qdwzpgw45dtl/spack-src/
	destination: /tmp/pkrastev/spack-stage/spack-stage-ncl-6.6.2-b44qiftbaxe2uwrgav54qdwzpgw45dtl/spack-src/triangle_src
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/ncl/set_spack_config.patch
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/ncl/ymake.patch
==> Ran patch() for ncl
==> ncl: Executing phase: 'install'
==> ncl: Successfully installed ncl-6.6.2-b44qiftbaxe2uwrgav54qdwzpgw45dtl
  Fetch: 0.08s.  Build: 39m 57.25s.  Total: 39m 57.33s.
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ncl-6.6.2-b44qiftbaxe2uwrgav54qdwzpgw45dtl
==> Updating view at /builds/pkrastev/Spack/spack/var/spack/environments/wrf/.spack-env/view
==> Warning: Skipping external package: intel-oneapi-mpi@2021.8.0%intel@2021.8.0~external-libfabric~generic-names~ilp64 build_system=generic arch=linux-rocky8-icelake/xfvjrn2
```

## List the installed packages

```
[wrf] [pkrastev@builds01 spack]$ spack find
==> In environment wrf
==> Root specs
intel-oneapi-mpi@2021.8.0

-- no arch / gcc@8.5.0 ------------------------------------------
cairo@1.16.0%gcc@8.5.0

-- no arch / intel@2021.8.0 -------------------------------------
hdf5@1.12%intel@2021.8.0 +cxx+fortran+hl+threadsafe  libpng@1.6.37%intel@2021.8.0  ncview@2.1.8%intel@2021.8.0    netcdf-fortran@4.6.0%intel@2021.8.0  wrf@4.4%intel@2021.8.0
jasper@1.900.1%intel@2021.8.0                        ncl@6.6.2%intel@2021.8.0      netcdf-c@4.9.0%intel@2021.8.0  wps@4.3.1%intel@2021.8.0             xz@5.4.2%intel@2021.8.0

==> Installed packages
-- linux-rocky8-icelake / gcc@8.5.0 -----------------------------
cairo@1.16.0

-- linux-rocky8-icelake / intel@2021.8.0 ------------------------
autoconf@2.69                       diffutils@3.8       hdf5@1.12.2                libmd@1.0.4           libxml2@2.10.1     ncview@2.1.8            py-pip@22.2.2           wps@4.3.1
automake@1.16.5                     esmf@8.3.1          inputproto@2.3.2           libpng@1.6.37         libxmu@1.1.2       netcdf-c@4.9.0          py-setuptools@65.5.0    wrf@4.4
bdftopcf@1.0.5                      expat@2.4.8         intel-oneapi-mpi@2021.8.0  libpthread-stubs@0.4  libxpm@3.5.12      netcdf-fortran@4.6.0    py-wheel@0.37.1         xcb-proto@1.14.1
berkeley-db@18.1.40                 findutils@4.9.0     jasper@1.900.1             libsigsegv@2.13       libxrender@0.9.10  ninja@1.11.1            python@3.10.8           xerces-c@3.2.3
berkeley-db@18.1.40                 flex@2.6.3          kbproto@1.0.7              libsm@1.2.3           libxt@1.1.5        openssl@1.1.1s          readline@8.1.2          xextproto@7.3.0
bison@3.8.2                         font-util@1.3.2     krb5@1.19.3                libtirpc@1.2.6        m4@1.4.19          openssl@1.1.1s          readline@8.1.2          xproto@7.0.31
bzip2@1.0.8                         fontconfig@2.13.94  libaec@1.0.6               libtool@2.4.7         makedepend@1.0.5   parallel-netcdf@1.12.3  renderproto@0.11.1      xtrans@1.3.5
bzip2@1.0.8                         fontsproto@2.1.3    libbsd@0.11.5              libx11@1.7.0          meson@0.63.3       pcre2@10.39             sqlite@3.39.4           xz@5.4.2
ca-certificates-mozilla@2022-10-11  freetype@2.11.1     libffi@3.4.2               libxau@1.0.8          mkfontdir@1.0.7    perl@5.36.0             tar@1.34                zlib@1.2.13
ca-certificates-mozilla@2022-10-11  gdbm@1.23           libfontenc@1.1.3           libxaw@1.0.13         mkfontscale@1.1.2  perl@5.36.0             tcsh@6.24.00            zlib@1.2.13
cmake@3.24.3                        gdbm@1.23           libice@1.0.9               libxcb@1.14           nasm@2.15.05       pigz@2.7                time@1.9                zstd@1.5.2
cmake@3.24.3                        gettext@0.21.1      libiconv@1.16              libxdmcp@1.1.2        ncl@6.6.2          pixman@0.40.0           udunits@2.2.28
curl@7.85.0                         glib@2.74.1         libiconv@1.16              libxext@1.3.3         ncurses@6.3        pkgconf@1.8.0           util-linux-uuid@2.38.1
diffutils@3.8                       gperf@3.1           libjpeg-turbo@2.1.3        libxfont@1.5.2        ncurses@6.3        pkgconf@1.8.0           util-macros@1.19.3
==> 110 installed packages
```

## Use WRF/ WPS

Once the environment is installed, WRF and WPS (and any other packages from the environment, such as `ncview`), are available on the `PATH`, e.g.,:

```bash
[wrf] [pkrastev@builds01 spack]$ which wrf.exe
/builds/pkrastev/Spack/spack/var/spack/environments/wrf/.spack-env/view/main/wrf.exe
```