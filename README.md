# Computer Vision Data Packer (cvdpack)

A tool to quantize and (optionally) video-compress your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos.

Reduce your dataset size by up to 90+%, with minimal changes in groundtruth accuracy!

:warning: This is an alpha release. Assume it might corrupt your data :warning:

**Make a backup of your data, and doublecheck your experimental results are not changed by cvdpack**

### Getting Started


##### Installation

Required: you must have `ffmpeg` installed an in your PATH. uv/pip will not install this currently. Choose one:
```bash
conda install ffmpeg
sudo apt install ffmpeg libx265-dev
brew install ffmpeg
# windows - TBD?
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

##### Example Commands:

You may need to `mkdir data/` before running the commands below. 
Use --verbose to see per-video output, or --debug to see per-frame. 
All commands should be usable with `uvx cvdpack` instead of `python -m cvdpack`

Pack a single tartanair scene locally. (Remove --subset to do the full thing)
```bash
# pack - runtime ~31sec, size reduced from 8.2G to 1.2G
python -m cvdpack.main pack_dataset --input data/TartanAir/ --output data/TartanAir_packed/ --config presets/tartanair.json --tmp_folder tmp/ --n_workers 20 --subset scene=abandonedfactory

# unpack - runtime ~28sec
python -m cvdpack.main unpack_dataset --input data/TartanAir_packed --output data/TartanAir_unpacked --n_workers 20 --tmp_folder tmp/ --subset scene=abandonedfactory
```

Pack/unpack all of TartanAir on a SLURM cluster (command shows works for princeton-vl's cluster; customize for your cluster)
```bash
# Pack the dataset - expected runtime ~
screen python -m cvdpack.main pack_dataset --input /n/fs/circuitnn/datasets/TartanAir --output /n/fs/scratch/$USER/data/TartanAir_packed --config presets/tartanair.json --tmp_folder /scratch/$USER/cvdpack_tmp/ --parallel_mode slurm --n_workers 200 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403

# unpack: expected runtime ~
screen python -m cvdpack.main unpack_dataset --input /n/fs/scratch/$USER/data/TartanAir_packed --output /n/fs/scratch/$USER/data/TartanAir_unpacked --tmp_folder /scratch/$USER/cvdpack_tmp/ --parallel_mode slurm --n_workers 100 --slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403
```

Partially pack/unpack tartanair (e.g just npys -> pngs, or just pngs -> mkvs, or mkvs -> pngs, or pngs -> npys). These can be run in sequence. 
```bash
python -m cvdpack.main pack_dataset --input data/TartanAir/ --output data/TartanAir_partialpack/ --config presets/tartanair.json --steps quantize --n_workers 20
python -m cvdpack.main pack_dataset --input data/TartanAir_partialpack/ --output data/TartanAir_pack/ --steps pack_video --n_workers 20
python -m cvdpack.main unpack_dataset --input data/TartanAir_pack/ --output data/TartanAir_partialunpack/ --steps unpack_video --n_workers 20
python -m cvdpack.main unpack_dataset --input data/TartanAir_partialunpack/ --output data/TartanAir_unpacked/ --steps unquantize --n_workers 20
```

Pack individual videos in TartanAir, step by step
```bash
# pack depth/flow into pngs (quantization)
python -m cvdpack.main pack_frames --input data/TartanAir/abandonedfactory/Hard/P000/flow/{frame:06d}_{framenext:06d}_flow.npy --output pngs/flow/{frame:06d}.png --to_dtype uint16 --quantize_method LINEAR --min_orig_val -150 --max_orig_val 150
python -m cvdpack.main pack_frames --input data/TartanAir/abandonedfactory/Hard/P000/depth/{frame:06d}_left_depth.png --output pngs/depth/{frame:06d}_left_depth.png --to_dtype float32 --quantize_method INV --min_orig_val 0.5 --max_orig_val 1000 --out_of_bounds_method nan

# pack pngs into mkv (video compression)
python -m cvdpack.main pack_frames --input pngs/flow/{frame:06d}.png --output vids/flow.mkv
python -m cvdpack.main pack_frames --input pngs/depth/{frame:06d}_left_depth.png --output vids/depth.mkv

# unpack mkv back into png (video decompression)
python -m cvdpack.main unpack_frames --input vids/flow.mkv --output pngs_unpacked/flow/{frame:06d}.png
python -m cvdpack.main unpack_frames --input vids/depth.mkv --output pngs_unpacked/depth/{frame:06d}_left_depth.png

# unpack png into depth/flow npys (unquantization)
python -m cvdpack.main unpack_frames --input pngs_unpacked/depth/{frame:06d}_left_depth.png --output unpack/depth/{frame:06d}_left_depth.npy --to_dtype float32 --quantize_method inv --min_orig_val 0.5 --max_orig_val 1000
python -m cvdpack.main unpack_frames --input pngs_unpacked/flow/{frame:06d}.png --output unpack/flow/{frame:06d}_{framenext:06d}_flow.npy --to_dtype uint16 --quantize_method linear --min_orig_val -150 --max_orig_val 150
```

Infinigen video scene
```
```

### User Guide

### Acknowledgement

This tool depends heavily on the incredible contributions of https://ffmpeg.org/ and https://opencv.org/

### Hypothetical TODOs:

I have no particular intention to continue adding features to this project. 

However, potential ideas would include:
- [ ] Allow scp-style prefixes to input and/or output path, in which case we read/write from remotes in a streaming fashion
- [ ] Use gpu accelerated decoders?
- [ ] Pack non-video framesets as compressed&chunked h5 (?) arrays
- [ ] Store stereo datasets efficiently by storing only left-frame info + sparse rightframe info
- [ ] Sbatch script which loads a dataset for you on job startup
- [ ] Dataloader which handles png->npy unpacking at runtime based on json config
