# Ollama

This shows how to run [ollama](https://github.com/ollama/ollama) on the FASRC
clusters.

1. Request an interactive job on a GPU partition
   ```
   salloc --partition gpu_test --gres=gpu:1 --mem-per-cpu 2G -c 4 --time 01:00:00
   ```

2. Download the [ollama Docker
   container](https://hub.docker.com/r/ollama/ollama) as a singularity container
   ```
   singularity pull docker://ollama/ollama
   ```
3. You will need two shells (i.e., 2 terminal/Putty windows)

   i. On terminal 1:
   ```
   singularity shell --nv ollama_latest.sif
   Singularity> export OLLAMA_HOST=localhost:8888
   Singularity> ollama serve
   ```
   ii. On terminal 2:
   ```
   # go to the same compute node where you are running ollama
   ssh nodename

   singularity shell --nv ollama_latest.sif
   Singularity> export OLLAMA_HOST=localhost:8888
   Singularity> ollama run llama3
   >>> Send a message (/? for help)
   ```
   Type your question and hit enter.

Example output of terminal 1:

```bash
[jharvard@holylogin04 ~]$ salloc --partition gpu_test --gres=gpu:1 --mem-per-cpu 2G -c 4 --time 01:00:00
salloc: Pending job allocation 34910069
salloc: job 34910069 queued and waiting for resources
salloc: job 34910069 has been allocated resources
salloc: Granted job allocation 34910069
salloc: Waiting for resource configuration
salloc: Nodes holygpu7c26105 are ready for job
[jharvard@holygpu7c26105 ~]$ singularity pull docker://ollama/ollama
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
28.2MiB / 28.2MiB [===================================================================================================] 100 % 74.4 MiB/s 0s
300.5MiB / 300.5MiB [=================================================================================================] 100 % 74.4 MiB/s 0s
35.4MiB / 35.4MiB [===================================================================================================] 100 % 74.4 MiB/s 0s
INFO:    Extracting OCI image...
INFO:    Inserting Singularity configuration...
INFO:    Creating SIF file...
[jharvard@holygpu7c26105 ~]$ singularity shell --nv ollama_latest.sif
Singularity> export OLLAMA_HOST=localhost:8888
Singularity> ollama serve
Couldn't find '/n/home01/jharvard/.ollama/id_ed25519'. Generating new private key.
Your new public key is:

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIQvB+nEomVHkitfKoqo7YXS1FwdB+aIwFJwMxgRwGML

2024/06/05 20:17:23 routes.go:1007: INFO server config env="map[OLLAMA_DEBUG:false OLLAMA_FLASH_ATTENTION:false OLLAMA_HOST: OLLAMA_KEEP_ALIVE: OLLAMA_LLM_LIBRARY: OLLAMA_MAX_LOADED_MODELS:1 OLLAMA_MAX_QUEUE:512 OLLAMA_MAX_VRAM:0 OLLAMA_MODELS: OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:1 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:*] OLLAMA_RUNNERS_DIR: OLLAMA_TMPDIR:]"
time=2024-06-05T20:17:23.754Z level=INFO source=images.go:729 msg="total blobs: 0"
time=2024-06-05T20:17:23.766Z level=INFO source=images.go:736 msg="total unused blobs removed: 0"
time=2024-06-05T20:17:23.773Z level=INFO source=routes.go:1053 msg="Listening on 127.0.0.1:8888 (version 0.1.40)"
time=2024-06-05T20:17:23.773Z level=INFO source=payload.go:30 msg="extracting embedded files" dir=/tmp/ollama1939682727/runners
time=2024-06-05T20:17:28.036Z level=INFO source=payload.go:44 msg="Dynamic LLM libraries [cpu_avx cpu_avx2 cuda_v11 rocm_v60002 cpu]"
time=2024-06-05T20:17:28.348Z level=INFO source=types.go:71 msg="inference compute" id=GPU-7aeed640-592c-3abf-3200-6a5724f43c81 library=cuda compute=8.0 driver=12.4 name="NVIDIA A100-SXM4-40GB MIG 3g.20gb" total="19.5 GiB" available="19.3 GiB"
```

Example output of terminal 2:

```bash
[jharvard@holylogin04 ~]$ ssh holygpu7c26105
[jharvard@holygpu7c26105 ~]$ singularity shell --nv ollama_latest.sif
Singularity> export OLLAMA_HOST=localhost:8888'
> ^C
Singularity> export OLLAMA_HOST=localhost:8888
Singularity> ollama run llama3
pulling manifest
pulling 6a0746a1ec1a... 100% ▕███████████████████████████████████████████████████████████████████████████▏ 4.7 GB
pulling 4fa551d4f938... 100% ▕███████████████████████████████████████████████████████████████████████████▏  12 KB
pulling 8ab4849b038c... 100% ▕███████████████████████████████████████████████████████████████████████████▏  254 B
pulling 577073ffcc6c... 100% ▕███████████████████████████████████████████████████████████████████████████▏  110 B
pulling 3f8eb4da87fa... 100% ▕███████████████████████████████████████████████████████████████████████████▏  485 B
verifying sha256 digest
writing manifest
removing any unused layers
success
>>> when will humans be able to teleport?
Teleportation, in the sense of moving an object or person from one location to another without crossing the space in between, is
still purely theoretical and has not been achieved scientifically. While there have been some experiments with quantum teleportation
(a process that transfers information about a particle's state rather than the particle itself), these are extremely limited and do
not involve macroscopic objects like humans.

As for human teleportation, it is currently considered a science fiction concept. There are many scientific and technological
challenges to overcome before we can even think about teleporting humans:

1. **Quantum entanglement**: To teleport an object or person, you would need to create a quantum connection between the two
locations, which is extremely difficult to achieve.
2. **Particle acceleration**: You'd need to accelerate particles to incredible speeds (close to the speed of light) to transfer
information about the person's state.
3. **Information storage and transmission**: Storing and transmitting vast amounts of information about an individual's physical
state, including their entire body, would require enormous computational power.
4. **Stability and control**: Maintaining stability and control during teleportation would be a significant challenge, as any errors
or disruptions could have disastrous consequences.

Given the current state of our understanding of quantum mechanics, relativity, and technology, it's unlikely that human teleportation
will become possible in the near future. However, scientists continue to explore new ways to manipulate matter at the atomic level,
which might lead to breakthroughs in fields like quantum computing or advanced transportation methods.

In summary, while teleportation remains a fascinating concept, it is not currently scientifically feasible for humans. However,
ongoing research and advancements in physics and technology may eventually lead to innovative solutions that could revolutionize our
understanding of space and time.

>>> /bye
Singularity>
```

