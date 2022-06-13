# jax_diffusion

An implementation of diffusion in jax.

## installation

install with conda (cpu only):

``` shell
conda env create -f environment.yml
conda activate jax_diffusion
```

or install with docker (gpu only):

* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run jax_diffusion
```

