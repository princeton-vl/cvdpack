set -e

rm -rf data/integration_test_tmp*/

# LOSSLESS
/usr/bin/time -o data/time_pack_lossless.txt uv run cvdpack pack --input data/TartanAir/ --output data/TartanAir_packed_lossless/ --config presets/tartanair_floatingpoint.json --tmp_folder data/integration_test_tmp1/ --n_workers 10 --subset scene=abandonedfactory vid=P000
/usr/bin/time -o data/time_unpack_lossless.txt uv run cvdpack unpack --input data/TartanAir_packed_lossless --output data/TartanAir_unpacked_lossless --n_workers 10 --tmp_folder data/integration_test_tmp3/ --subset scene=abandonedfactory vid=P000

du -h --max-depth 4 data/TartanAir/ > data/size_raw.txt
du -h --max-depth 4 data/TartanAir_packed_lossless/ > data/size_packed_lossless.txt
du -h --max-depth 4 data/TartanAir_unpacked_lossless/ > data/size_unpacked_lossless.txt

uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/image_{cam}/{frame:06d}_{cam}.{ext} --output data/TartanAir_unpacked_lossless/abandonedfactory/Hard/P000/image_{cam}/{frame:06d}_{cam}.{ext} --error
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} --output data/TartanAir_unpacked_lossless/abandonedfactory/Hard/P000/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} --error
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/flow/{frame:06d}_{f2:06d}_flow.npy --output data/TartanAir_unpacked_lossless/abandonedfactory/Hard/P000/flow/{frame:06d}_{f2:06d}_flow.npy --error

# LOSSY

CVDPACK_MINOR_VIDEO_ERROR_CODECS=1 /usr/bin/time -o data/time_pack_lossy.txt uv run cvdpack pack --input data/TartanAir/ --output data/TartanAir_packed_lossy/ --config presets/tartanair_quantized.json --tmp_folder data/integration_test_tmp2/ --n_workers 10 --subset scene=abandonedfactory vid=P000
/usr/bin/time -o data/time_unpack_lossy.txt uv run cvdpack unpack --input data/TartanAir_packed_lossy --output data/TartanAir_unpacked_lossy --n_workers 10 --tmp_folder data/integration_test_tmp4/ --subset scene=abandonedfactory vid=P000

du -h --max-depth 4 data/TartanAir_packed_lossy/ > data/size_packed_lossy.txt
du -h --max-depth 4 data/TartanAir_unpacked_lossy/ > data/size_unpacked_lossy.txt

uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/image_{cam}/{frame:06d}_{cam}.{ext} --output data/TartanAir_unpacked_lossy/abandonedfactory/Hard/P000/{gt_type}_{cam}/{frame:06d}_{cam}.{ext} --error
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/seg_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} --output data/TartanAir_unpacked_lossy/abandonedfactory/Hard/P000/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} --error
uv run -m cvdpack.checkdiff --input data/TartanAir/abandonedfactory/Hard/P000/flow/{frame:06d}_{f2:06d}_flow.npy --output data/TartanAir_unpacked_lossy/abandonedfactory/Hard/P000/flow/{frame:06d}_{f2:06d}_flow.npy --error --atol 0.01
