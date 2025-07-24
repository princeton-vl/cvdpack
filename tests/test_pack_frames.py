import numpy as np

from cvdpack.pack_frames import (
    LinearQuantizeIntPacker,
    InvQuantizeInt16Packer,
    CheckBoundsPacker,
    F32As2Int16ReinterpretPacker,
    F16ToInt16ReinterpretPacker,
)


def test_linear_pack_method():
    np.random.seed(42)
    data = np.random.uniform(1.0, 50.0, (32, 32)).astype(np.float32)
    min_val, max_val = 0.0, 100.0
    to_dtype = np.uint16

    packer = LinearQuantizeIntPacker(min_val, max_val, data.dtype, to_dtype)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed)

    prec = (max_val - min_val) / np.iinfo(to_dtype).max
    np.testing.assert_allclose(unpacked, data, atol=prec * 1.01)


def test_inv_pack_method():
    np.random.seed(42)
    data = np.random.uniform(0.5, 20.0, (32, 32)).astype(np.float32)
    min_val, max_val = 0.1, 50.0
    to_dtype = np.uint16

    packer = InvQuantizeInt16Packer(min_val, max_val, data.dtype, to_dtype)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed)

    np.testing.assert_allclose(unpacked, data, rtol=1e-2)


def test_checkbounds_pack_method():
    np.random.seed(42)
    data = np.random.randint(0, 1000, (32, 32), dtype=np.uint32)
    min_val, max_val = 0, 1000
    to_dtype = np.uint16

    packer = CheckBoundsPacker(min_val, max_val, to_dtype, data.dtype)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed)

    np.testing.assert_array_equal(unpacked, data.astype(to_dtype))


def test_onechannel_f32_as_2int16_pack_method():
    np.random.seed(42)
    min_val, max_val = -100.0, 100.0
    data = np.random.uniform(min_val, max_val, (32, 32)).astype(np.float32)

    packer = F32As2Int16ReinterpretPacker(scalar=1.0)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed).squeeze(-1)

    np.testing.assert_allclose(unpacked, data, rtol=1e-6)


def test_multichannel_to_f16_as_int16_pack_method():
    np.random.seed(42)
    data = np.random.uniform(-1000, 1000, (32, 32, 3)).astype(np.float32)
    _min_val, _max_val = 0.0, 1.0

    packer = F16ToInt16ReinterpretPacker(scalar=1.0)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed)

    rdiffs = np.abs(unpacked - data) / data
    largest_rdiff = np.max(rdiffs)
    print(f"largest_rdiff: {largest_rdiff}")

    np.testing.assert_allclose(unpacked, data, rtol=1e-3)


def test_nan_handling():
    np.random.seed(42)
    data = np.random.uniform(1.0, 10.0, (16, 16)).astype(np.float32)
    data[::4, ::4] = np.nan

    min_val, max_val = 1.0, 10.0
    to_dtype = np.uint16

    packer = LinearQuantizeIntPacker(min_val, max_val, data.dtype, to_dtype)
    packed = packer.pack(data)
    unpacked = packer.unpack(packed)

    nan_mask = np.isnan(data)
    assert np.isnan(unpacked[nan_mask]).all()
    np.testing.assert_allclose(unpacked[~nan_mask], data[~nan_mask], rtol=1e-3)
