# Installing `glmnet` and `glmtrans`

```bash
# request an interactive job
salloc -p test -t 2:00:00 --mem 8000

# load modules
module load R/4.2.0-fasrc01
module load gcc/12.1.0-fasrc01

# set environmental variables
unset R_LIBS_SITE
mkdir -p $HOME/apps/R/4.2.0
export R_LIBS_USER=$HOME/apps/R/4.2.0:$R_LIBS_USER

# setup compiler for glmnet
mkdir -p $HOME/.R
echo -e '## C++ flags \nCXX14=g++ \nCXX14PICFLAGS=-fPIC' >> $HOME/.R/Makevars

# open R shell
R
```

Inside R shell

```
> install.packages("glmnet")
> install.packages("glmtrans")
> library(glmnet)
Loading required package: Matrix
Loaded glmnet 4.1-4
> library(glmtrans)
> quit()
```

Outside R shell

```bash
# rename Makevars so it won't interfere with future package installs
mv $HOME/.R/Makevars $HOME/.R/Makevars_glmnet_glmtrans
```

