# *******************************************************************************
# *******************************************************************************
# DO NOT EDIT THIS SCRIPT !!!
# *******************************************************************************
# *******************************************************************************
# This script (`setup_heavyAI`) is sourced directly into the main slurm script
# that is run when the batch job starts up. 
#
# There are some helpful Bash functions that are made available to this script
# that encapsulate commonly used routines when initializing a web server:
#
#   - find_port
#       Find available port in range [$1..$2]
#       Default: 2000 65535
#
#   - create_passwd
#       Generate random alphanumeric password with $1 characters
#       Default: 32
#
# We **MUST** supply the following environment variables in this
# `setup_heavyAI.sh` script so that a user can connect back to the web server when
# it is running (case-sensitive variable names):
#
#   - $host (already defined earlier, so no need to define again)
#       The host that the web server is listening on
#
#   - $port
#       The port that the web server is listening on
#
#   - $password
#       The plain text password used to authenticate to the web server with

# Source in all the helper functions
source_helpers () {
  # Generate random integer in range [$1..$2]
  random_number () {
    shuf -i ${1}-${2} -n 1
  }
  export -f random_number

  port_used_python() {
    python -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_python3() {
    python3 -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_nc(){
    nc -w 2 "$1" "$2" < /dev/null > /dev/null 2>&1
  }

  port_used_lsof(){
    lsof -i :"$2" >/dev/null 2>&1
  }

  port_used_bash(){
    local bash_supported=$(strings /bin/bash 2>/dev/null | grep tcp)
    if [ "$bash_supported" == "/dev/tcp/*/*" ]; then
      (: < /dev/tcp/$1/$2) >/dev/null 2>&1
    else
      return 127
    fi
  }

  # Check if port $1 is in use
  port_used () {
    local port="${1#*:}"
    local host=$((expr "${1}" : '\(.*\):' || echo "localhost") | awk 'END{print $NF}')
    local port_strategies=(port_used_nc port_used_lsof port_used_bash port_used_python port_used_python3)

    for strategy in ${port_strategies[@]};
    do
      $strategy $host $port
      status=$?
      if [[ "$status" == "0" ]] || [[ "$status" == "1" ]]; then
        return $status
      fi
    done

    return 127
  }
  export -f port_used

  # Find available port in range [$2..$3] for host $1
  # Default: [2000..65535]
  find_port () {
    local host="${1:-localhost}"
    local port=$(random_number "${2:-2000}" "${3:-65535}")
    while port_used "${host}:${port}"; do
      port=$(random_number "${2:-2000}" "${3:-65535}")
    done
    echo "${port}"
  }
  export -f find_port

  # Wait $2 seconds until port $1 is in use
  # Default: wait 30 seconds
  wait_until_port_used () {
    local port="${1}"
    local time="${2:-30}"
    for ((i=1; i<=time*2; i++)); do
      port_used "${port}"
      port_status=$?
      if [ "$port_status" == "0" ]; then
        return 0
      elif [ "$port_status" == "127" ]; then
         echo "commands to find port were either not found or inaccessible."
         echo "command options are lsof, nc, bash's /dev/tcp, or python (or python3) with socket lib."
         return 127
      fi
      sleep 0.5
    done
    return 1
  }
  export -f wait_until_port_used

  # Generate random alphanumeric password with $1 (default: 8) characters
  create_passwd () {
    tr -cd 'a-zA-Z0-9' < /dev/urandom 2> /dev/null | head -c${1:-8}
  }
  export -f create_passwd
}
export -f source_helpers

source_helpers

# Export the module function if it exists
[[ $(type -t module) == "function" ]] && export -f module

# Find available port to run server on
port=$(find_port localhost 7000 11000 )
export MAPD_TCP_PORT=${port}

port=$(find_port localhost 7000 11000 )
export MAPD_HTTP_PORT=${port}

port=$(find_port localhost 7000 11000 )
export MAPD_CALCITE_PORT=${port}

## this is the one we need to expose to the proxy
port=$(find_port localhost 7000 11000 )
export MAPD_WEB_PORT=${port}

# Generate SHA1 encrypted password (requires OpenSSL installed)
SALT="$(create_passwd 16)"
password="$(create_passwd 16)"
PASSWORD_SHA1="$(echo -n "${password}${SALT}" | openssl dgst -sha1 | awk '{print $NF}')"

export HEAVYAIBASE=/scratch/$USER/$SLURM_JOB_ID

# if you would like to provide the path of your database, set it in "myfolder" variable
#myfolder=
#mkdir -p $myfolder
#[ -d $myfolder ] && export HEAVYAIBASE=$myfolder

echo "HEAVYAIBASE: " $HEAVYAIBASE

# Create HEAVYAI_BASE as explained in
# https://docs.heavy.ai/installation-and-configuration/installation/upgrading-omnisci/upgrading-omnisci-1#essential-changes-for-release-6.0-of-heavy.ai-compared-to-omnisci
mkdir -p ${HEAVYAIBASE}/var/lib/heavyai/

## For some reason if I use the config.py file to configure the notebook, the package nb_conda_kernels does not seem to work correctly.
#  I will temporarily pass those attributes as command line arguments 
export MY_PASSWD=sha1:${SALT}:${PASSWORD_SHA1}
export MY_BASEURL=/node/${host}/${port}/

