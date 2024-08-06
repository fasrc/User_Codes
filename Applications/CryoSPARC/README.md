# CryoSPARC

[See FASRC Docs](https://docs.rc.fas.harvard.edu/kb/cryosparc/)

## Configure CryoSPARC Master
The master program is called with the `cryosparcm` command documented [here](https://guide.cryosparc.com/setup-configuration-and-management/management-and-monitoring/cryosparcm). 
The major mechanism for customizing the behavior of `cryosparcm` is the config file located in `cryosparc_master/config.sh`.
A basic `config.sh` might look like the following
```bash
export CRYOSPARC_LICENSE_ID="________-____-____-____-____________"
export CRYOSPARC_DB_PATH="_________________________________________/cryosparc_database"
export CRYOSPARC_DEVELOP=false
export CRYOSPARC_INSECURE=false
export CRYOSPARC_CLICK_WRAP=true
export CRYOSPARC_MASTER_HOSTNAME=holy_______.rc.fas.harvard.edu
export CRYOSPARC_BASE_PORT=____
```
Containing the license, path to the MongoDB datbase, master hostname, and the base tcp port. 

## Installing CryoSPARC on Cannon

There is a provided [configure.sh script](configure.sh) to get you up and running fast, or proceed interactively if preferred. 

The install script is based on the [instructions](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc) found in the CryoSPARC guide. 
This script should typically be run __on a GPU node__ so that the correct CUDA modules are loaded and functioning. 
Sequentially, this script does the following steps:
 1) Set environment variables particular to this install
 2) Remove any potential old files that will cause the install to fail
 3) Load the appropriate CUDA related modules
 4) Make the installation directories
 5) [Download](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#use-curl-to-download-the-two-files-into-tarball-archives) the CryoSPARC Master and Worker binaries
 6) Unpack the binaries
 7) [Install](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#install-the-cryosparc_master-package) the master binary
 8) [Install](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#gpu-worker-node-cryosparc-installation) the worker binary
 9) [Add](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#create-the-first-user) an initial user account
 10) [Add](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc#connect-a-cluster-to-cryosparc) the [cluster configuration](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/cryosparc-cluster-integration-script-examples#slurm) for running jobs on Cannon
 
## Running cryosparc on Cannon
To run CryoSPARC on Cannon, several things need to happen. 
First an open port on the login node in the range `7000-11000` needs to be identified (see [here](https://docs.rc.fas.harvard.edu/kb/jupyter-notebook-server-on-cluster/)). 
Next a CPU node should be allocated using the `SLURM` job scheduler. 
The CryoSPARC Master configuration file, `$INSTALL_DIR/cryosparc_master/config.sh` needs to be modified to reflect the available port and node hostname.
Then the `cryosparcm` master process can be started on the CPU node. 
In order to adapt the MongoDB database to a potentially new port number, the following commands need to be run once the Master has started. 

```shell
cryosparcm fixdbport
cryosparcm restart
```

Then the user must set up an `SSH` tunnel from their local machine through the login node to the compute node. Note that this can also be done using a VDI session as your login node.  The VDI instance will have better performance.  Just substitute the name of the VDI node for the login node name.

```shell
ssh -NL port:compute_node:port user@holylogin.rc.fas.harvard.edu
```

Once authenticated, the CryoSPARC webapp should be viewable at `http://localhost:port`. 


## Mostly Automated CryoSPARC Connection
Because the launch process is tedious and error prone, I have automated much of it in [a shell function](bashrc_additions) which the user can add to their cluster `.bashrc`.

This uses a temporary file `CRYOSPARC_CONNECTION_SCRIPT` in order to monitor the progress of the `cryosparcm` launch. 
This script is written out into the user's home directory after the server initializes. 


To connect from your local terminal, add the following to your local `.bashrc`. 

```bash
connect_to_cryosparc()
{
    LOGIN=$holylogin.rc.fas.harvard.edu
    USERNAME=______________
    CONNECTION_SCRIPT=".cryosparc_connection_script.sh"
    rsync $USERNAME@:~/$CONNECTION_SCRIPT . 
    chmod +x $CONNECTION_SCRIPT
    ./$CONNECTION_SCRIPT
}
```

Using these two shell functions, CryoSPARC is launched in two steps:
 1) On the login node, type `launchcryosparc`
 2) Once, the launch function returns, type `connect_to_cryosparc` on the local machine

