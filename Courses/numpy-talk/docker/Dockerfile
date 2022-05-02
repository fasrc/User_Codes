# We use Mamba rather than conda since it's faster and uses the newer `conda-forge` channel.
FROM condaforge/mambaforge
SHELL ["/bin/bash", "--login", "-c"]

# Disable interactive prompts when installing packages
ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y build-essential pkg-config libomp-dev

RUN mamba install -y python=3.10
RUN pip install --upgrade pip
RUN mamba install -c conda-forge jupyter -y

COPY . /src/
WORKDIR /src
RUN pip install -r setup-requirements.txt
RUN pip install .
RUN chmod a+rx /src/bin/main

COPY docker/entrypoint.sh /usr/bin 
RUN chmod a+rx /usr/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

