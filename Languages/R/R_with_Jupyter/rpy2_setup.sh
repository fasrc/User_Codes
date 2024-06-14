#!/bin/bash

#The script makes the apps directory in user's $HOME. It takes a
#command line argument for the version of the R module that needs to
#be loaded.

RVERSION=$1
if [ -z "$RVERSION" ] || [ $# -lt 1 ]
then
    echo "This script requires R module version as its 1st command line argument"
    exit
fi

echo "R Version is $RVERSION"

DIRNAME="$HOME/apps/R_$RVERSION"
echo "Directory name is $DIRNAME"

if [ ! -d "$HOME/apps" ]; then
    mkdir $HOME/apps
fi

if [ ! -d "$DIRNAME" ]; then
    mkdir $DIRNAME
else
    echo "$DIRNAME already exists"
fi

export R_LIBS_USER=${HOME}/apps/R_${RVERSION}
