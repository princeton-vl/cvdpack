# Computer Vision Data Packer (cvdpack)

A tool to quantize and (optionally) video-compress your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos.

Reduce your dataset size by up to 90+%, with minimal changes in groundtruth accuracy!

:warning: This is an alpha release. Assume it might corrupt your data :warning:

**Make a backup of your data, and doublecheck your experimental results are not changed by cvdpack**

### Getting Started


##### Installation

```bash
git clone https://github.com/princeton-vl/cvdpack.git
cd cvdpack

# use your own conda if you want
conda create --name myenv python=3.11
conda activate myenv


pip install -e .
```

##### Example Commands:

One line commands for TartanAir (using pre-written config)
```bash
# pack
python -m cvdpack.main pack dataset --input TartanAir/ --output TartanAir_packed/ --config presets/tartanair.json --tmp_folder tmp/ --parallel_mode multiprocess --n_jobs 10

#unpack
python -m cvdpack.main unpack dataset --input TartanAir_packed --output TartanAir_unpacked
```

Pack tartanair on the ionic cluster using /scratch tmp

```bash
python -m cvdpack.main pack dataset --input /n/fs/circuitnn/datasets/TartanAir --output /n/fs/scratch/$USER/TartainAirPacked --config presets/tartanair.json --tmp_folder /scratch/$USER/cvdpack_tmp/ --parallel_mode slurm --n_jobs 10 --slurm_args account=pvl nodelist=nodelist=node007,node[020-026],node[101-104],node403 cpus=4
```

TartanAir, step by step
```bash

# pack depth/flow into pngs
python -m cvdpack.main pack frameset --input TartanAir/abandonedfactory/Hard/P000/flow/{frame:06d}_{framenext:06d}_flow.npy --output pngs/flow/{frame:06d}.png --to_dtype uint16 --quantize_method SYM_SQRT --min_orig_val -512 --max_orig_val 512
python -m cvdpack.main pack frameset --input TartanAir/abandonedfactory/Hard/P000/depth/{frame:06d}_left_depth.png --output pngs/depth/{frame:06d}_left_depth.png --to_dtype float32 --quantize_method INV --min_orig_val 0.5 --max_orig_val 1000 --out_of_bounds_method nan

# pack pngs into mkv
python -m cvdpack.main pack frameset --input pngs/flow/{frame:06d}.png --output vids/flow.mkv
python -m cvdpack.main pack frameset --input pngs/depth/{frame:06d}_left_depth.png --output vids/depth.mkv

# unpack mkv back into png
python -m cvdpack.main unpack frameset --input vids/flow.mkv --output pngs_unpacked/flow/{frame:06d}.png
python -m cvdpack.main unpack frameset --input vids/depth.mkv --output pngs_unpacked/depth/{frame:06d}_left_depth.png

# unpack png into depth/flow npys
python -m cvdpack.main unpack frameset --input pngs_unpacked/depth/{frame:06d}_left_depth.png --output unpack/depth/{frame:06d}_left_depth.npy --to_dtype float32 --quantize_method inv --min_orig_val 0.5 --max_orig_val 1000
python -m cvdpack.main unpack frameset --input pngs_unpacked/flow/{frame:06d}.png --output unpack/flow/{frame:06d}_{framenext:06d}_flow.npy --to_dtype uint16 --quantize_method linear --min_orig_val -512 --max_orig_val 512
```

Infinigen video scene
```
```

### User Guide

### Hypothetical TODOs:

I have no particular intention to continue adding features to this project. 

However, potential ideas would include:
- [ ] Allow scp-style prefixes to input and/or output path, in which case we read/write from remotes in a streaming fashion
- [ ] Pack non-video framesets as compressed&chunked h5 (?) arrays
- [ ] Store stereo datasets efficiently by storing only left-frame info + sparse rightframe info
- [ ] Sbatch script which loads a dataset for you on job startup
- [ ] Dataloader which handles png->npy unpacking at runtime based on json config
