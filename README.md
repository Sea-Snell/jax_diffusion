# jax_diffusion

An implementation of diffusion in jax.

## installation

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate jax_diffusion
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate jax_diffusion
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**install with docker (gpu only):**
* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run jax_diffusion
```

