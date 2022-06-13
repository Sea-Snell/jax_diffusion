# syntax=docker/dockerfile:1
FROM condaforge/miniforge3:latest
SHELL ["/bin/bash", "--login", "-c"]
WORKDIR /app/
CMD ["bash"]

# install basics
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/google/jax

# install python packages
COPY environment.yml .
RUN conda env create -f environment.yml
RUN rm -rf environment.yml
RUN echo "conda activate jax_diffusion" >> ~/.bashrc
RUN source ~/.bashrc
