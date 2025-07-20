# Computer Vision Data Packer (uvx cvdpack)

A tool to reorganize and save space on your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos. Reduce your dataset size by up to 90+%, with minimal changes in groundtruth accuracy!

:warning: Make a backup of your data, and doublecheck your experimental results are not changed by uvx cvdpack :warning:

Currently, this tool has been minimally tested to work with the following dataset(s)
- TartanAir
- TODO: test more. 

### Installation

Required: you must have `ffmpeg` installed and in your PATH. Currently I have not configured uv/pip to install this for you (TODO)

If ffmpeg is not already installed, choose an install option:
```bash
conda install ffmpeg
sudo apt install ffmpeg libx265-dev
brew install ffmpeg
# windows - TODO?
```

Then, install uv: [instructions](https://docs.astral.sh/uv/getting-started/installation/)

You can now run `uvx cvdpack` as shown below. You do not need to manually install the tool to use the cli. 

##### Optional: install cvdpack package

Installing the python package is only necessary if you want to use the python interface. choose one:
```bash
uv pip install cvdpack
pip install cvdpack
```

### Example Commands:

Cvdpack works for many dataset - see `--help` for all options and `--presets`, or use your own `--config myfile.json`

Commands will print very little output unless using -v or -d. 

##### Pack/unpack tartanair scene locally. 
Commands shown are for a single scene and video, remove --subset to do the full thing
```bash
uvx cvdpack pack --input data/TartanAir/ --output data/TartanAir_packed/ --config presets/tartanair_quantized.json --tmp_folder data/tmp/ --n_workers 10 --subset scene=abandonedfactory vid=P000 -v

uvx cvdpack unpack --input data/TartanAir_packed --output data/TartanAir_unpacked --n_workers 10 --tmp_folder data/tmp/ --subset scene=abandonedfactory vid=P000 -v
```
Runtime for one scene is approx 31sec and 28sec respectively on a AMD EPYC 7713P 64-core machine.
Filesizes are approx 8.6GB for the raw abandonedfactory/Hard/P000 scene, 526M for the packed version (94% savings)

##### Pack/unpack all of TartanAir on a SLURM cluster 
Commands shown work for princeton-vl's cluster, you will need to customize the paths and slurm args for own cluster.
```bash
screen uvx cvdpack pack --input /n/fs/circuitnn/datasets/TartanAir --output /n/fs/scratch/$USER/data/TartanAir_packed --config presets/tartanair_quantized.json --tmp_folder /scratch/$USER/uvx cvdpack_tmp/ --parallel_mode slurm --n_workers 200 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403

screen uvx cvdpack unpack --input /n/fs/scratch/$USER/data/TartanAir_packed --output /n/fs/scratch/$USER/data/TartanAir_unpacked --tmp_folder /scratch/$USER/uvx cvdpack_tmp/ --parallel_mode slurm --n_workers 200 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403
```

##### Partially pack/unpack TartanAir 

e.g just npys -> pngs, or just pngs -> mkvs, or mkvs -> pngs, or pngs -> npys. These can be run in sequence. 

```bash
uvx cvdpack pack --input data/TartanAir/ --output data/TartanAir_partialpack/  --steps quantize --n_workers 10 --cpus_per_worker 4 --config presets/tartanair_quantized.json
uvx cvdpack pack --input data/TartanAir_partialpack/ --output data/TartanAir_packed/ --steps pack_video --n_workers 10 --cpus_per_worker 4
uvx cvdpack unpack --input data/TartanAir_packed/ --output data/TartanAir_partialunpack/ --steps unpack_video --n_workers 10 --cpus_per_worker 4 
uvx cvdpack unpack --input data/TartanAir_partialunpack/ --output data/TartanAir_unpacked/ --steps unquantize --n_workers 10 --cpus_per_worker 4
```
For a single scene (abandonedfactory/Hard/P000):
- Runtimes are approx 34sec, 58sec, 12sec, 23sec respectively on a AMD EPYC 7713P 64-core machine.
- Result sizes are approx TODO, TODO, TODO, TODO respectively.

##### Reorganize a dataset
```bash
uvx cvdpack copy --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} --output data/TartanAir_split/{scene}/{split}_{vid}/{cam}/{gt_type}/{frame:04d}.{ext}
```

##### Extract a subset of a dataset
```bash
uvx cvdpack copy --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} --output data/TartanAir_split/{} --subset scene=abandonedfactory split=Hard vid=P000,P001 gt_type=image,depth cam=left
```
Note: currently struggles to do the whole dataset for some dataset layouts e.g. TartanAir which stores many gt types in the same folder (flow and mask).

### Acknowledgement

This tool depends heavily on the incredible contributions of https://ffmpeg.org/ and https://opencv.org/

### Contributing:

Pull requests welcome!
- No need to send PRs for typos or code style.
- Please describe what you intended to achieve
- show example commands of what it does, including timing and file sizes
- if your command uses public datasets as a test, please link to the dataset. 

Use github issues for feature requests or bugs. 

Further development of this project might occur through community contributions, but is not a high priority for the main author(s). 

##### Developer install

```bash
git clone https://github.com/princeton-vl/cvdpack.git
cd cvdpack
uv pip install -e .[dev]
```

You should then run all the example commands via `uv run` instead of `uvx`

##### Unit tests
```bash
uv run pytest tests/
```

##### Difference checker tool:

Run tartanair pack and unpack, then choose one:

```bash
# all of TartanAir, except flow, that one uses a different template :/
uv run -m cvdpack.checkdiff --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{cam}.{ext} --output data/TartanAir_unpacked/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext}

# TartanAir, all depth
uv run -m cvdpack.checkdiff --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}__{cam}.{ext} --output data/TartanAir_unpacked/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{frame:06d}_{gt_type}.{ext} --subset gt_type=depth

# TartanAir, all flow images
uv run -m cvdpack.checkdiff --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}__{cam}.{ext} --output data/TartanAir_unpacked/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{frame:06d}_{gt_type}.{ext} --subset gt_type=flow

# TartanAir, single depth image
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/depth_left/000000_left_depth.npy --output data/TartanAir_unpacked/abandonedfactory/Hard/P000/depth_left/000000_left_depth.npy

# TartanAir, single flow image
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/flow_left/000000_000001_flow.npy --output data/TartanAir_unpacked/abandonedfactory/Hard/P000/flow_left/000000_000001_flow.npy


```

Note: you can also run these with concrete single image paths and not use --subset


##### TODOs

Planned:
- [ ] More presets/ .json files for common datasets
- [ ] Add support for sintel/flyingthings .flo .disp .pfm etc
- [ ] Allow pack resolution or res multiplier to be specified in config, enforce this during pack / unpack
- [ ] Allow scp-style prefixes to input and/or output path, in which case we read/write from remotes in a streaming fashion

No particular roadmap or intention to complete:
- [ ] Provide a default dataloader which handles any packed dataset w.r.t cvdpack.json
    - [ ] Primary task: Dataload and unpack frames from a packed version of the dataset
    - [ ] Dataload from mkv version of the dataset ??
- [ ] Use gpu accelerated ffmpeg decoders for faster unpack at startup? are there any lossless ones?
- [ ] Pack non-video framesets as compressed & chunked h5 (?) arrays
- [ ] Store surface normals / unit sphere data as 2 angles, instead of 3 coords for 2dof. Use 2xuint16 quant or 2xfloat16 packing
- [ ] Store stereo datasets efficiently by storing only left-frame info + sparse rightframe info
- [ ] Sbatch script which loads a dataset for you on job startup
- [ ] Dataloader which handles png->npy unpacking at runtime, with mapping based on json config
