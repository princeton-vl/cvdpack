import numpy as np

from cvdpack.main import PackMethod, img_orig_to_pack, img_pack_to_orig


def test_linear_pack_method():
    np.random.seed(42)
    data = np.random.uniform(1.0, 50.0, (32, 32)).astype(np.float32)
    min_val, max_val = 0.0, 100.0
    to_dtype = np.uint16

    packed = img_orig_to_pack(data, to_dtype, PackMethod.LINEAR, min_val, max_val)
    unpacked = img_pack_to_orig(packed, min_val, max_val, data.dtype, PackMethod.LINEAR)

    prec = (max_val - min_val) / np.iinfo(to_dtype).max
    np.testing.assert_allclose(unpacked, data, atol=prec * 1.01)


def test_inv_pack_method():
    np.random.seed(42)
    data = np.random.uniform(0.5, 20.0, (32, 32)).astype(np.float32)
    min_val, max_val = 0.1, 50.0
    to_dtype = np.uint16

    packed = img_orig_to_pack(data, to_dtype, PackMethod.INV, min_val, max_val)
    unpacked = img_pack_to_orig(packed, min_val, max_val, data.dtype, PackMethod.INV)

    np.testing.assert_allclose(unpacked, data, rtol=1e-2)


def test_checkbounds_pack_method():
    np.random.seed(42)
    data = np.random.randint(0, 1000, (32, 32), dtype=np.uint32)
    min_val, max_val = 0, 1000
    to_dtype = np.uint16

    packed = img_orig_to_pack(data, to_dtype, PackMethod.CHECKBOUNDS, min_val, max_val)
    unpacked = img_pack_to_orig(
        packed, min_val, max_val, data.dtype, PackMethod.CHECKBOUNDS
    )

    np.testing.assert_array_equal(unpacked, data.astype(to_dtype))


def test_onechannel_f32_as_2int16_pack_method():
    np.random.seed(42)
    data = np.random.uniform(-100.0, 100.0, (32, 32)).astype(np.float32)
    min_val, max_val = -100.0, 100.0
    to_dtype = np.uint16

    packed = img_orig_to_pack(
        data, to_dtype, PackMethod.ONECHANNEL_F32_AS_2INT16, min_val, max_val
    )
    unpacked = img_pack_to_orig(
        packed, min_val, max_val, data.dtype, PackMethod.ONECHANNEL_F32_AS_2INT16
    )

    np.testing.assert_allclose(unpacked, data, rtol=1e-6)


def test_multichannel_to_f16_as_int16_pack_method():
    np.random.seed(42)
    data = np.random.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    min_val, max_val = 0.0, 1.0
    to_dtype = np.uint16

    packed = img_orig_to_pack(
        data, to_dtype, PackMethod.MULTICHANNEL_TO_F16_AS_INT16, min_val, max_val
    )
    unpacked = img_pack_to_orig(
        packed, min_val, max_val, data.dtype, PackMethod.MULTICHANNEL_TO_F16_AS_INT16
    )

    np.testing.assert_allclose(unpacked, data, rtol=1e-3)


def test_nan_handling():
    np.random.seed(42)
    data = np.random.uniform(1.0, 10.0, (16, 16)).astype(np.float32)
    data[::4, ::4] = np.nan

    min_val, max_val = 1.0, 10.0
    to_dtype = np.uint16

    packed = img_orig_to_pack(data, to_dtype, PackMethod.LINEAR, min_val, max_val)
    unpacked = img_pack_to_orig(packed, min_val, max_val, data.dtype, PackMethod.LINEAR)

    nan_mask = np.isnan(data)
    assert np.isnan(unpacked[nan_mask]).all()
    np.testing.assert_allclose(unpacked[~nan_mask], data[~nan_mask], rtol=1e-3)
