#!/bin/bash --login

set -euo pipefail
# /src/bin/main -N 100 -K 2 -T 100 -p "$@"
jupyter notebook --ip 0.0.0.0 --port=8888 --allow-root --no-browser