# Computer Vision Data Packer (uvx cvdpack)

A tool to reorganize and save space on your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos.

Reduce your dataset size by up to 90+%, with minimal changes in groundtruth accuracy!

:warning: Make a backup of your data, and doublecheck your experimental results are not changed by uvx cvdpack :warning:

### Installation

Required: you must have `ffmpeg` installed and in your PATH. Currently I have not configured uv/pip to install this for you (TODO)

If ffmpeg is not already installed, choose an install option:
```bash
conda install ffmpeg
sudo apt install ffmpeg libx265-dev
brew install ffmpeg
# windows - TODO?
```

Install uv ([detailed instructions](https://docs.astral.sh/uv/getting-started/installation/))
```bash
#Mac/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### Optional: install cvdpack package

You can use `uvx` or `uv tool run` as shown below WITHOUT installing cvdpack first. 

Installing the python package is only necessary if you want to use the python interface
```bash
uv pip install cvdpack
```

##### Non-uv installation

Follow ffmpeg instructions as shown above, and create your own conda environmnent if you wish. Then:
```bash
pip install cvdpack
```

##### Developer install

```bash
git clone https://github.com/princeton-vl/cvdpack.git
cd cvdpack
uv pip install -e .[dev]
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

### TODO:

Planned:
- [ ] More presets/ .json files for common datasets
- [ ] Add support for sintel/flyingthings .flo .disp .pfm etc
- [ ] Allow pack resolution or res multiplier to be specified in config, enforce this during pack / unpack
- [ ] Allow scp-style prefixes to input and/or output path, in which case we read/write from remotes in a streaming fashion

No particular roadmap or intention to complete:
- [ ] Provide a default dataloader which handles any uvx cvdpack.json
    - [ ] Load from png version of the dataset
    - [ ] Load from mkv version of the dataset ??
- [ ] Use gpu accelerated video decoders?
- [ ] Pack non-video framesets as compressed & chunked h5 (?) arrays
- [ ] Store surface normals / unit sphere data as 2 angles, instead of 3 coords for 2dof. Use 2xuint16 quant or 2xfloat16 packing
- [ ] Store stereo datasets efficiently by storing only left-frame info + sparse rightframe info
- [ ] Sbatch script which loads a dataset for you on job startup
- [ ] Dataloader which handles png->npy unpacking at runtime, with mapping based on json config
