# Simplicits

![Teaser](assets/teaser.gif)

This codebase implements [Simplicits](https://research.nvidia.com/labs/toronto-ai/simplicits/) allows simulating a mesh, Gaussian Splat, or a NeRF by learning a reduced deformation basis. I implemented the paper and then compared my results and implmentation with the original implementation, thus this codebase has a few differences, these end up having qualitative and speed improvements to the original codebase.

This codebase was also my submission for an assignment at UofT, which required us to implement a paper discussed in class. This was for a course called Topics in Computer Graphics: Physics Based Animation CSC2521.

## Summary of Differences

This codebase has a few differences from the original codebase. Here is a summary of those:

- A different way of training the model.

When minimizing elastic energy functions, models often collapse to trivial solutions where all handles encode constant weights, effectively limiting the model to rigid motions. The way this was handled in the original paper was by adding a very large loss term based on heuristics to enforce the outputs of the neural net to be orthonormal. This seemed like a hack to me, so I changed the way the model is trained and enforced this by the design of the training. I summarize these changes [here](https://rishitdagli.com/projects/simplicits/oth.pdf). Alongside not having to rely on a hyperparameter/heuristic, this also **qualitatively improves the results**.

- A different implementation of the loss calculation.

In my experiments on the Gaussian Splat for the lego dozer (on RTX 4090), training a model with the original implmentation of loss from `kaolin` takes ~15.96 ms per iteration, training it for 20k iterations gets us to ~319.64 s. However, with the new implementation of the loss, training the same model takes ~1.97 ms per iterations, training it for 20k iterations gets us to ~39.56 s, **an ~87.6% speedup in training**.

- Simulating NeRFs.

I noticed the original codebase did not have a way to train Simplicits for NeRFs. While this codebase does allow training to simulate NeRFs from Nerfstudio, simulating the NeRFs from just the forward deformation map is a bit cumbersome. So for right now, during rendering, I render point clouds.

- Density and Opacity thresholds.

The original codebase had a constant opacity threshold of 0.35 to choose points for 3DGS, which does not work for all kinds of scenes. While, I was implementing this, I was testing on a bunch of different objects and I eneded up implementing a way to automatically choose a good density threshold for NeRFs and a opacity threshold for 3DGS. This **qualitatively improves the results**.

## Installation

I recommend using a virtual environment to install the required packages. You can install the required packages by running the following command assuming you are in the root of the repository,

```bash
pip install -r requirements.txt
pip install -e .
```

Instead of installing the packages from the `requirements.txt` file, I would recommend:

- Install `torch==2.4.1` (any `torch>2.0.0` should work) and `torchvision==0.19.1`
- Install `ninja`
- Install `nerfstudio` (I used the codebase at [this commit](https://github.com/nerfstudio-project/nerfstudio/commit/555d5540086cc6e85717be6b07cc37d5d07af893)) by following the instructions [here](https://docs.nerf.studio/quickstart/installation.html)
- Install `kaolin` by following the instructions [here](https://kaolin.readthedocs.io/en/latest/notes/installation.html)
- Install the rest of the packages from the `requirements.txt` file
- Finally, install this codebase by running `pip install -e .`

## Usage

The codebase has two main commands, `ns-train-simplicits` to train a model and `ns-viewer-simplicits` to simulate and render the object. Both of these commands, work on meshes, 3DGS, and NeRFs.

On a high level, these commands can be used as follows:

```
usage: ns-train-simplicits [-h] [OPTIONS]

╭─ options ─────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                           │
│ --base-type STR         Type of the scene we are working with, either: "mesh",    │
│                         "nerf", or "gs" (default: mesh)                           │
│ --original-geometry PATH                                                          │
│                         Path to the original geometry. (1) If base_type is        │
│                         "mesh", this should be a .obj file. (2) If base_type is   │
│                         "nerf", this should be a .yaml config file. (3) If        │
│                         base_type is "gs", this should be a .ply file. (default:  │
│                         .)                                                        │
│ --density-threshold {None}|FLOAT                                                  │
│                         Geometry Parameters - Density threshold for the NeRF      │
│                         model or the opacity threshold for the GaussianSplat. If  │
│                         not provided, we will compute an adaptive threshold.      │
│                         (default: None)                                           │
│ --output-path {None}|PATH                                                         │
│                         Path where we need to save the weights (default: None)    │
│ --device {None}|STR|DEVICE                                                        │
│                         Training parameters - device to train on (default: cuda)  │
│ --orig-loss {None,True,False}                                                     │
│                         Training parameters - should we use the original loss     │
│                         function in the paper, i recommend setting this to False  │
│                         (default: False)                                          │
│ --iters INT             Training parameters - number of iterations of simplicits  │
│                         (default: 10000)                                          │
│ --num-samples INT       Training parameters - number of points to sample from the │
│                         geometry (default: 1000000)                               │
│ --cubature-pts INT      Training parameters - number of cubature points to        │
│                         sample, default from paper suppplementary (default: 2000) │
│ --handles INT           Training parameters - number of handles for the model for │
│                         simulation, default from paper supplementary for popular  │
│                         splats (default: 40)                                      │
│ --layers INT            Training parameters - number of layers in the MLP model   │
│                         (default: 9)                                              │
│ --batch-size INT        Training parameters - batch size (default: 16)            │
│ --start-lr FLOAT        Training parameters - starting learning rate (default:    │
│                         0.001)                                                    │
│ --end-lr FLOAT          Training parameters - ending learning rate (default:      │
│                         0.001)                                                    │
│ --optimizer STR         Training parameters - either "custom" or "adam" (my       │
│                         experiments indicate decreasing quality and decreasing    │
│                         training time from left to right). (default: custom)      │
│ --soft-youngs-modulus FLOAT                                                       │
│                         Physical material properties - Young's modulus i.e.       │
│                         stiffness (default: 100000.0)                             │
│ --poisson-ratio FLOAT   Physical material properties - Poisson's ratio i.e. ratio │
│                         of lateral strain to longitudinal strain (default: 0.45)  │
│ --rho FLOAT             Physical material properties - Density (default: 100)     │
│ --approx-volume FLOAT   Physical material properties - Approximate volume of the  │
│                         object (default: 1)                                       │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

```
usage: ns-viewer-simplicits [-h] [OPTIONS]

╭─ options ─────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                           │
│ --base-type STR         Type of the scene we are working with, either: "mesh",    │
│                         "nerf", or "gs" (default: mesh)                           │
│ --original-geometry PATH                                                          │
│                         Path to the original geometry. (1) If base_type is        │
│                         "mesh", this should be a .obj file. (2) If base_type is   │
│                         "nerf", this should be a .yaml config file. (3) If        │
│                         base_type is "gs", this should be a .ply file. (default:  │
│                         .)                                                        │
│ --density-threshold {None}|FLOAT                                                  │
│                         Geometry Parameters - Density threshold for the NeRF      │
│                         model or the opacity threshold for the GaussianSplat. If  │
│                         not provided, we will compute an adaptive threshold.      │
│                         (default: None)                                           │
│ --model-path PATH       Path to the trained model (default: model.safetensors)    │
│ --device {None}|STR|DEVICE                                                        │
│                         Training parameters - device to train on (default: cuda)  │
│ --handles INT           Training parameters - number of handles for the model for │
│                         simulation, default from paper supplementary for popular  │
│                         splats (default: 40)                                      │
│ --layers INT            Training parameters - number of layers in the MLP model   │
│                         (default: 9)                                              │
│ --num-samples INT       Training parameters - number of points to sample from the │
│                         geometry (default: 1000000)                               │
│ --num-steps INT         Number of simulation steps (default: 100)                 │
│ --soft-youngs-modulus FLOAT                                                       │
│                         Physical material properties - Young's modulus i.e.       │
│                         stiffness (default: 100000.0)                             │
│ --poisson-ratio FLOAT   Physical material properties - Poisson's ratio i.e. ratio │
│                         of lateral strain to longitudinal strain (default: 0.45)  │
│ --rho FLOAT             Physical material properties - Density (default: 100)     │
│ --approx-volume FLOAT   Physical material properties - Approximate volume of the  │
│                         object (default: 1)                                       │
│ --floor-height FLOAT    Scene parameters - Floor height (default: -0.8)           │
│ --floor-penalty FLOAT   Scene parameters - Penalty for the floor (default: 1000)  │
│ --gravity [FLOAT [FLOAT ...]]                                                     │
│                         Scene parameters - Gravity (default: 0 9.8 0)             │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

I now show a few examples that reproduce the examples from the original paper or the original codebase.

### Mesh Example

First train a model to simulate an example mesh, or directly download the pre-trained model for this mesh from [here](https://github.com/Rishit-dagli/simplicits-nerfstudio/releases/download/v0.1.0/mesh.safetensors).

```bash
ns-train-simplicits --base-type mesh \
    --original-geometry assets/mesh/fox.obj \
    --output-path mesh.safetensors \
    --device cuda \
    --iters 10000 \
    --num-samples 1000000 \
    --cubature-pts 2048 \
    --handles 5 \
    --layers 9 \
    --batch-size 16 \
    --start-lr 1e-3 \
    --end-lr 1e-5
```

Now simulate the mesh using the trained model, the following command opens up a viewer usually on http://localhost:7007/ (or some other port if that is occupied). Follow the instructions on [using the viewer](#using-the-viewer).

```bash
ns-viewer-simplicits --base-type mesh \
    --original-geometry assets/mesh/fox.obj \
    --model-path mesh.safetensors \
    --device cuda \
    --handles 5 \
    --layers 9 \
    --num-samples 1000000 \
    --num-steps 100
```

## Gaussian Splat Example

Download the pre-trained model for the Gaussian Splat from [here](https://github.com/Rishit-dagli/simplicits-nerfstudio/releases/download/v0.1.0/3dgs_dozer.ply). 

<details><summary>Optionally train a splat yourself</summary>

You can optionally also train the model yourself with nerfstudio. Start by downloading the synthetic nerf dataset by running,

```bash
mkdir -p data/nerfstudio/
cd data/nerfstudio/
curl -L -o nerf-synthetic-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nguyenhung1903/nerf-synthetic-dataset
unzip nerf-synthetic-dataset.zip
rm nerf-synthetic-dataset.zip
cd ../..
```

Then train a 3DGS,

```bash
ns-train splatfacto-big \
    --pipeline.model.use-bilateral-grid True \
    --data data/nerfstudio/nerf_synthetic/lego blender-data
```

Finally, make a `ply` export of the splat, make sure to replace the config file with the one you get from the training,

```bash
ns-export gaussian-splat \
    --load-config outputs/lego/splatfacto/2024-12-22_232014/config.yml \
    --output-dir exports/splat/
```

You can now train simplicits on the `.ply` file you get from the export by changing the argument `--original-geometry`.
</details>

First, train a model to simulate an example Gaussian Splat, or directly download the pre-trained model for this Gaussian Splat from [here](https://github.com/Rishit-dagli/simplicits-nerfstudio/releases/download/v0.1.0/gs.safetensors).

```bash
ns-train-simplicits --base-type gs \
    --original-geometry 3dgs_dozer.ply \
    --output-path gs.safetensors \
    --device cuda \
    --iters 20000 \
    --num-samples 2048 \
    --cubature-pts 2048 \
    --handles 40 \
    --layers 10 \
    --batch-size 16 \
    --start-lr 1e-3 \
    --end-lr 1e-3 \
    --soft-youngs-modulus 21000 \
    --poisson-ratio 0.45 \
    --rho 100 \
    --approx-volume 3
```

Now simulate the Gaussian Splat using the trained model, the following command opens up a viewer usually on http://localhost:7007/ (or some other port if that is occupied). Follow the instructions on [using the viewer](#using-the-viewer).

```bash
ns-viewer-simplicits --base-type gs \
    --original-geometry 3dgs_dozer.ply \
    --model-path gs.safetensors \
    --device cuda \
    --handles 40 \
    --layers 10 \
    --num-samples 2048 \
    --num-steps 100 \
    --soft-youngs-modulus 15000 \
    --poisson-ratio 0.45 \
    --rho 100 \
    --approx-volume 3
```

Since this example is oriented differently make sure to set the Floor Axis to Z and Gravity Direction to Z, and then click on reset simulation in the viewer.

## NeRF Example

Download the pre-trained instant-ngp model from [here](https://github.com/Rishit-dagli/simplicits-nerfstudio/releases/download/v0.1.0/nerf_dozer.zip) and unzip it.

<details><summary>Optionally train a NeRF yourself</summary>

You can optionally also train the model yourself with nerfstudio. Start by downloading the synthetic nerf dataset by running,

```bash
mkdir -p data/nerfstudio/
cd data/nerfstudio/
curl -L -o nerf-synthetic-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nguyenhung1903/nerf-synthetic-dataset
unzip nerf-synthetic-dataset.zip
rm nerf-synthetic-dataset.zip
cd ../..
```

Then train a NeRF, make sure to replace the `--original-geometry` argument with the path to the config file you get from the training,

```bash
ns-train instant-ngp-bounded \
    --data="data/nerfstudio/nerf_synthetic/lego/transforms_train.json" \
    --steps-per-save=1000 \
    --max-num-iterations=16500
```
</details>

First, train a model to simulate an example NeRF, or directly download the pre-trained model for this NeRF from [here](https://github.com/Rishit-dagli/simplicits-nerfstudio/releases/download/v0.1.0/nerf.safetensors).

```bash
ns-train-simplicits --base-type nerf \
    --original-geometry outputs/lego/instant-ngp-bounded/2024-12-22_111233/config.yml \
    --density-threshold 0.8 \
    --output-path nerf.safetensors \
    --device cuda \
    --iters 20000 \
    --num-samples 2048 \
    --cubature-pts 2048 \
    --handles 40 \
    --layers 10 \
    --batch-size 16 \
    --start-lr 1e-4 \
    --end-lr 1e-5 \
    --soft-youngs-modulus 15000 \
    --poisson-ratio 0.45 \
    --rho 100 \
    --approx-volume 3
```

Now simulate the NeRF using the trained model, the following command opens up a viewer usually on http://localhost:7007/ (or some other port if that is occupied). Follow the instructions on [using the viewer](#using-the-viewer).

```bash
ns-viewer-simplicits --base-type nerf \
    --original-geometry outputs/lego/instant-ngp-bounded/2024-12-22_111233/config.yml \
    --density-threshold 0.8 \
    --model-path nerf.safetensors \
    --device cuda \
    --handles 40 \
    --layers 10 \
    --num-samples 100000 \
    --num-steps 100 \
    --soft-youngs-modulus 15000 \
    --poisson-ratio 0.45 \
    --rho 100 \
    --approx-volume 3
```

Since this example is oriented differently make sure to set the Floor Axis to Z and Gravity Direction to Z, and then click on reset simulation in the viewer.

## Using the Viewer

In this short section, I will explain how to use the viewer to simulate objects. The viewer builds on top of [nerfstudio viser](https://viser.studio/latest/) and I recommend taking a look at the documentation. In this section, I explain how to use the changes this project makes to the viewer.

At the start, the viewer looks like this,

![Viewer](assets/viewer_rest.png)

The "Time" folder on the right is used to play the simulation or see individual 3D frames of the simulation. The "Canonical" checkbox always stops the simulation and shows the original rest geometry.

The "Scene Parameters" folder on the right is used to change the properties of the scene, like external forces and floor properties. Make sure to click on "Reset Simulation" after changing anything from this folder.

<details><summary>Example</summary>

![Viewer Simualation](assets/viewer_scene.png)
</details>

The "Point Cloud" and "Render Tracks" checkboxes are used to show the point cloud as well as the 4D tracks of some parts of the object.

<details><summary>Example</summary>

![Viewer Point Cloud](assets/viewer_pc_track.png)
</details>

## Tips

I share some tips on running the code and reproducing results.

### on installing required packages

- Installing Nerfstudio, especially on HPC systems can be tricky. I recommend installing `open3d`, and `tiny-cuda-nn` before installing Nerfstudio separately and installing it from source. I also recommend building these packages on the same GPU you plan to run it on.
- When you install PyTorch, especially on HPC systems, you will often end up with at least two versions of CUDA: one which is installed when you install PyTorch and is not a full version of CUDA and the other which is in the system. I highly recommend manually installing the same version of CUDA as in the system that PyTorch automatically installs.
- I use virtualenv and use the `requirements.txt` file to install the required packages.

### on compute

- I have tested the code on A100 - 80 GB and RTX 4090 GPUs. I have not tested the code on other GPUs, however, I expect the code to work on other GPUs with atleast 24 GB VRAM.
- For just training Simplicits you should not need more than 16 GB VRAM.

### debugging

- If you see some jitter in the simulation, try reducing the FPS in the viewer.
- Rendering tracks can be slow, especially for large geometries, I recommend turning this off if you are running into performance issues.
- Very rarely when starting the viewer I get some error with Viser, however this is flaky, and is always fixed by simply rerunning the viewer command. I haven't been able to figure out why this happens yet.
