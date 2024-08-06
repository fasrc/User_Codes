Primary Author: Kevin Dalton

### What is CryoSPARC?
[CryoSPARC](https://guide.cryosparc.com/) is a closed source, commercially-developed piece software for analyzing single-particle cryoelectron microscopy data.
It supports CUDA based GPU-accelerated analysis through the PyCUDA library.
It consists of several applications which are bundled in two separate binary packages, termed
 - CryoSPARC Master (`cryosparcm`)
 - CryoSPARC Worker

The Master package is meant to use relative little compute resources, and at least some sysadmins seem to have decided to allow users to run this directly on login nodes. 
CryoSPARC Worker can be run on a separate node or the same node, but typically should have access to GPU compute resources. 
The worker nodes must have password-less SSH access to the master node as well as unfettered TCP on a number of ports. 
The authoritative list of requirements for installation can be found in the CryoSPARC [guide](https://guide.cryosparc.com/setup-configuration-and-management/cryosparc-installation-prerequisites). 

In addition to instantiating worker nodes and connecting them to the Master node, CryoSPARC can also be configured with a "Cluster Lane" which submits jobs via the SLURM job scheduler. 
This is the install strategy described in this document. 

### CryoSPARC Master
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

At the top level, `cryosparcm` is really a `Supervisor` based `shell` script which manages at least six different applications.
For instance, running `cryosparcm start` will bring up the following applications
 - app (cli)
 - command_core
 - command_rtp
 - command_vis
 - database (MongoDB)
 - webapp

As far as I can tell these mostly communicate with one another over `TCP`. 
The `TCP` ports used by each of the component programs are not individually configurable, but the base port to which the user connects is configurable in the `cryosparc_master/config.sh`. 
Of note, the hostname of the node running the master application is also typically hardcoded in this `config.sh` file. 
However, if this is left unset, it will take the hostname of the machine on which `cryosparcm start` is called. 

### Obtaining a CryoSPARC License
CryoSPARC is free for academic use. 
However, it does require a license.
You can [request a license](http://cryosparc.com/download) on the CryoSPARC webpage. 
The process of requesting a license is described in detailed [here](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/obtaining-a-license-id).

### Installing CryoSPARC on Cannon
This monolithic install script is based on the [instructions](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc) found in the CryoSPARC guide. 
This script should typically be run __on a GPU node__ so that the correct CUDA modules are loaded and functioning. 
Sequentially, this script does the following steps
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
 
 These steps can be run sequentially [with this configure script](configure.sh) or interactively if preferred. 

### Running cryosparc on Cannon
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


### Mostly Automated CryoSPARC Connection
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

