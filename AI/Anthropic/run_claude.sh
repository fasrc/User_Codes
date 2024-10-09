#!/bin/bash

# Source local variables
source local.conf

# Request an interactive job
salloc --partition=gpu_test --gres=gpu:1 --time=02:00:00 --mem=8G --cpus-per-task=2

# Source conda environment
mamba activate claude_env

# set SSL_CERT_FILE with system's certificate
export SSL_CERT_FILE='/etc/pki/tls/certs/ca-bundle.crt'

# run Claude example
python claude_quickstart.py
