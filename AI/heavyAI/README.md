# HeavyAI

In this document you will find instructions to submit a batch job that will
create access to HeavyAI without using Open OnDemand portal
(https://rcood.rc.fas.harvard.edu/).

You can still use [Open
OnDemand](https://rcood.rc.fas.harvard.edu/pun/sys/dashboard) if you prefer.

## What is HeavyAI?

See HeavyAI [website](https://www.heavy.ai/product/overview).

This software was formerly known as OmniSci.

## HeavyAI in the FASRC Cannon cluster

HeavyAI is implemented on the FASRC cluster using Singulaity. 

For details about Singularity see:

- [FASRC Singularity documentation](../../Singularity_Containers)
- [Singularity documentation](https://docs.sylabs.io/guides/latest/user-guide/introduction.html)

### Singularity images

HeavyAI singularity containers are stored in a cluster-wide location, meaning
that individual users **do not** have to copy the singularity images to use
them. Singularity images are located in:

```bash
/n/singularity_images/OOD/omnisci/
```

Each singularity container file is tagged with the HeavyAI (or formely OmniSci)
version:

```bash
[jharvard@holylogin01 ~]$ ls -lh  /n/singularity_images/OOD/omnisci/
total 4.1G
drwxr-xr-x. 2 root root 4.0K Mar  2  2020 current
-rwxr-xr-x. 1 root root 2.1G May 25  2023 heavyai-ee-cuda_v7.0.0.sif
-rwxr-xr-x. 1 root root 2.1G Jan 23 14:23 heavyai-ee-cuda_v7.2.2.sif
drwxr-xr-x. 2 root root 4.0K Jan 13  2021 v5.5.0
drwxr-xr-x. 2 root root 4.0K May 17  2021 v5.5.5
drwxr-xr-x. 2 root root 4.0K Sep 27  2021 v5.7.1
```

Singularity containers were pulled from [HeavyAI's
DockerHub](https://hub.docker.com/r/heavyai/heavyai-ee-cuda).

## Running HeavyAI

We recommend running HeavyAI on GPU partitions because HeavyAI was designed to
run in GPUs. For the specifics of each slurm partition, see [Cannon
partitions](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Slurm_partitions)
and [FASSE
partitions](https://docs.rc.fas.harvard.edu/kb/fasse/#SLURM_and_Partitions). 


To run HeavyAI, you will need two scripts:

* [`setup_heavyAI.sh`](setup_heavyAI.sh): sets up the network connections that
    you will need to access HeavyAI. You do not need to edit this file
* [`run_heavyAI.sh`](run_heavyAI.sh): slurm batch script that will:
  1. Request computing resources to run HeavyAI
  2. Source (i.e., executes) `setup_heavyAI.sh`
  3. Sets the working director as the variable `${HEAVYAIBASE}`
  4. Defines the HeavyAI singularity container to run
  5. Bind slurm variables to the Singularity container
  6. Bind HeavyAI `${HEAVYAI_BASE}`
  7. Print tunneling commands
  8. Run HeavyAI

You will have to edit in the `run_heavyAI.sh` script:

* `SBATCH` directives to suit your needs (e.g. time `-t`, number of cores `-c`, 
    amount of memory `--mem`)
* `container_image` depending on the version that you would like to use

We recommend carefully reading [HeavyAI hardware
recommendation](https://docs.heavy.ai/installation-and-configuration/system-requirements/hardware)
as they provide details about number of cores and RAM (memory) that you should
request. They also recommend using SSD storage, which is available through [local
scratch](https://docs.rc.fas.harvard.edu/kb/cluster-storage/#Local_per_node_Shared_Scratch_Storage).

You may request specific GPU cards on the FASRC clusters using the
`--constraint` flag. See our [Job
Contrainst](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Job_Constraints)
documentation for more details. 

**Step 1:** Submit batch job

```bash
[jharvard@holylogin01 heavyAI]$ sbatch run_heavyAI.sh
Submitted batch job 17242795
```

**Step 2:** Get contents of `heavyAI.out`

Once the job is running (note that your job may be in queue for a while until it
gets started), it will generate an output file `heavyAI.out`. The content of
`heavyAI.out` will have the `ssh` command for tunneling and the `localhost`
link. For example:

```bash
[jharvard@holylogin01 heavyAI]$ cat heavyAI.out
HEAVYAIBASE:  /scratch/jharvard/17242795

=====================================================================
execute ssh command from local computer:
ssh -NL 10495:holygpu8a22202:10495 jharvard@login.rc.fas.harvard.edu
=====================================================================

startheavy 1410071 running
Backend TCP:  localhost:10160
Backend HTTP: localhost:10782
Frontend Web: localhost:10495
Calcite TCP:  localhost:10720
- heavydb 1410256 started
- heavy_web_server 1410257 started
Rebrand migration: Added symlink from "/var/lib/heavyai/storage/mapd_catalogs" to "catalogs"
Rebrand migration: Added symlink from "/var/lib/heavyai/storage/mapd_data" to "data"
Rebrand migration: Added symlink from "/var/lib/heavyai/storage/mapd_export" to "export"
Rebrand migration: Added symlink from "/var/lib/heavyai/storage/mapd_log" to "log"
Rebrand migration: Added symlink from "/var/lib/heavyai/storage/catalogs/omnisci_system_catalog" to "system_catalog"
Rebrand migration completed
⇨ http server started on [::]:10495
Navigate to: http://localhost:10495
```

**Step 3:** Copy this line

```bash
ssh -NL 10495:holygpu8a22202:10495 jharvard@login.rc.fas.harvard.edu
```

**Step 4:** On your local machine, paste the `ssh` command. Type in your
password and two-factor authentication. It will look likes it hangs, but that's
ok. It should look similar to this:

```bash
[paula@mac ~]$ ssh -NL 10495:holygpu8a22202:10495 jharvard@login.rc.fas.harvard.edu
(jharvard@login.rc.fas.harvard.edu) Password:
(jharvard@login.rc.fas.harvard.edu) VerificationCode:

```

**Step 5:** Copy the url in the last line of `heavyAI.out`:

```bash
http://localhost:10495
```

**Step 6:** Paste the url on your browser. You will be able to connect to
HeavyAI.

**Note:** You will need to provide your own license key to use HeavyAI. FASRC
does not provide a license. You can request a free version on HeavyAI [downloads
page](https://www.heavy.ai/product/downloads). Note that the free license only
allows limited computational resources.

## Resources

* [HeavyAI Immerse documentation](https://docs.heavy.ai/immerse/introduction-to-immerse)
* [HeavyAI harware
    specs](https://docs.heavy.ai/installation-and-configuration/system-requirements/hardware)


