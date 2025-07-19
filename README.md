# Computer Vision Data Packer (cvdpack)

A tool to reorganize and save space on your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos.

Reduce your dataset size by up to 90+%, with minimal changes in groundtruth accuracy!

:warning: Make a backup of your data, and doublecheck your experimental results are not changed by cvdpack :warning:

## Installation

Required: you must have `ffmpeg` installed and in your PATH. Currently I have not configured uv/pip to install this for you (TODO)

If ffmpeg is not already installed, choose an install option:
```bash
conda install ffmpeg
sudo apt install ffmpeg libx265-dev
brew install ffmpeg
# windows - TODO?
```

Install from PyPi. Choose one:
```bash
uv pip install cvdpack
pip install cvdpaclk

# if you want to use --paralell_mode slurm
uv pip install cvdpack[slurm]
pip install cvdpack[slurm]
```

Install from source
```bash
git clone https://github.com/princeton-vl/cvdpack.git
cd cvdpack
pip install -e .
```

## Example Commands:

Please see `cvdpack --help` for all options!

Note: we use TartanAir as an example dataset, but cvdpack is not specific to TartanAir.

Commands will print very little output, unless they fail or you add -v or --debug

#### Pack/unpack tartanair scene locally. 
Commands shown are for a single scene and video, remove --subset to do the full thing
```bash
cvdpack pack_dataset --input data/TartanAir/ --output data/TartanAir_packed/ --config presets/tartanair.json --tmp_folder data/tmp/ --n_workers 10 --subset scene=abandonedfactory vid=P000 -v

cvdpack unpack_dataset --input data/TartanAir_packed --output data/TartanAir_unpacked --n_workers 10 --tmp_folder data/tmp/ --subset scene=abandonedfactory vid=P000 -v
```
Runtime for one scene is approx 31sec and 28sec respectively on a AMD EPYC 7713P 64-core machine.
Filesizes are approx 8.6GB for the raw abandonedfactory/Hard/P000 scene, 526M for the packed version (94% savings)

#### Pack/unpack all of TartanAir on a SLURM cluster 
Commands shown work for princeton-vl's cluster, you will need to customize the paths and slurm args for own cluster.
```bash
screen cvdpack pack_dataset --input /n/fs/circuitnn/datasets/TartanAir --output /n/fs/scratch/$USER/data/TartanAir_packed --config presets/tartanair.json --tmp_folder /scratch/$USER/cvdpack_tmp/ --parallel_mode slurm --n_workers 200 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403

screen cvdpack unpack_dataset --input /n/fs/scratch/$USER/data/TartanAir_packed --output /n/fs/scratch/$USER/data/TartanAir_unpacked --tmp_folder /scratch/$USER/cvdpack_tmp/ --parallel_mode slurm --n_workers 200 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403
```

#### Partially pack/unpack TartanAir 

e.g just npys -> pngs, or just pngs -> mkvs, or mkvs -> pngs, or pngs -> npys. These can be run in sequence. 

```bash
cvdpack pack_dataset --input data/TartanAir/ --output data/TartanAir_partialpack/  --steps quantize --n_workers 10 --cpus_per_worker 4 --config presets/tartanair.json
cvdpack pack_dataset --input data/TartanAir_partialpack/ --output data/TartanAir_packed/ --steps pack_video --n_workers 10 --cpus_per_worker 4
cvdpack unpack_dataset --input data/TartanAir_packed/ --output data/TartanAir_partialunpack/ --steps unpack_video --n_workers 10 --cpus_per_worker 4 
cvdpack unpack_dataset --input data/TartanAir_partialunpack/ --output data/TartanAir_unpacked/ --steps unquantize --n_workers 10 --cpus_per_worker 4
```
For a single scene (abandonedfactory/Hard/P000):
- Runtimes are approx 34sec, 58sec, 12sec, 23sec respectively on a AMD EPYC 7713P 64-core machine.
- Result sizes are approx TODO, TODO, TODO, TODO respectively.

#### Reorganize a dataset
```bash
cvdpack copy --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} --output data/TartanAir_split/{scene}/{split}_{vid}/{cam}/{gt_type}/{frame:04d}.{ext}
```

#### Extract a subset of a dataset
```bash
cvdpack copy --input data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} --output data/TartanAir_split/{} --subset scene=abandonedfactory split=Hard vid=P000,P001 gt_type=image,depth cam=left
```
Note: currently struggles to do the whole dataset for some dataset layouts e.g. TartanAir which stores many gt types in the same folder (flow and mask).

#### Pack individual videos in TartanAir, step by step
```bash
# pack depth/flow into pngs (quantization)
cvdpack pack_frames --input data/TartanAir/abandonedfactory/Hard/P000/flow/{frame:06d}_{framenext:06d}_flow.npy --output pngs/flow/{frame:06d}.png --to_dtype uint16 --pack_method LINEAR --min_orig_val -150 --max_orig_val 150
cvdpack pack_frames --input data/TartanAir/abandonedfactory/Hard/P000/depth/{frame:06d}_left_depth.png --output pngs/depth/{frame:06d}_left_depth.png --to_dtype float32 --pack_method INV --min_orig_val 0.5 --max_orig_val 1000 --out_of_bounds_method nan

# pack pngs into mkv (video compression)
cvdpack pack_frames --input pngs/flow/{frame:06d}.png --output vids/flow.mkv
cvdpack pack_frames --input pngs/depth/{frame:06d}_left_depth.png --output vids/depth.mkv

# unpack mkv back into png (video decompression)
cvdpack unpack_frames --input vids/flow.mkv --output pngs_unpacked/flow/{frame:06d}.png
cvdpack unpack_frames --input vids/depth.mkv --output pngs_unpacked/depth/{frame:06d}_left_depth.png

# unpack png into depth/flow npys (unquantization)
cvdpack unpack_frames --input pngs_unpacked/depth/{frame:06d}_left_depth.png --output unpack/depth/{frame:06d}_left_depth.npy --to_dtype float32 --pack_method inv --min_orig_val 0.5 --max_orig_val 1000
cvdpack unpack_frames --input pngs_unpacked/flow/{frame:06d}.png --output unpack/flow/{frame:06d}_{framenext:06d}_flow.npy --to_dtype uint16 --pack_method linear --min_orig_val -150 --max_orig_val 150
```

### Acknowledgement

This tool depends heavily on the incredible contributions of https://ffmpeg.org/ and https://opencv.org/

### TODO:

Planned:
- [ ] More presets/ .json files for common datasets
- [ ] Add support for sintel/flyingthings .flo .disp .pfm etc
- [ ] Allow pack resolution or res multiplier to be specified in config, enforce this during pack / unpack
- [ ] Allow scp-style prefixes to input and/or output path, in which case we read/write from remotes in a streaming fashion

No particular roadmap or intention to complete:
- [ ] Provide a default dataloader which handles any cvdpack.json
    - [ ] Load from png version of the dataset
    - [ ] Load from mkv version of the dataset ??
- [ ] Use gpu accelerated video decoders?
- [ ] Pack non-video framesets as compressed&chunked h5 (?) arrays
- [ ] Store stereo datasets efficiently by storing only left-frame info + sparse rightframe info
- [ ] Sbatch script which loads a dataset for you on job startup
- [ ] Dataloader which handles png->npy unpacking at runtime, with mapping based on json config
