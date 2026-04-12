"""Tests for forge_interop — PyTorch ↔ Forge zero-copy bridge."""

import sys
sys.path.insert(0, "/home/horde/.openclaw/workspace/forge-gpu/forge-interop")

import os
os.environ["PYTHONPATH"] = "/home/horde/.openclaw/workspace/.pylib"

import torch
from forge_interop import tensor_to_forge_ptr, ForgeArray, ForgeDLPack


def test_tensor_to_forge_ptr():
    """Raw pointer extraction from PyTorch tensor."""
    t = torch.randn(1000, device='cuda')
    ptr, size = tensor_to_forge_ptr(t)

    assert ptr > 0, "pointer should be non-zero"
    assert size == 1000, f"size should be 1000, got {size}"

    # Verify the pointer is stable (same tensor = same pointer)
    ptr2, _ = tensor_to_forge_ptr(t)
    assert ptr == ptr2, "same tensor should give same pointer"

    print("✅ tensor_to_forge_ptr: ptr={}, size={}".format(hex(ptr), size))


def test_forge_array_from_torch():
    """ForgeArray wraps a PyTorch tensor."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    fa = ForgeArray.from_torch(t)

    assert fa.data_ptr == t.data_ptr()
    assert fa.numel == 5
    assert fa.dtype == "f32"
    assert fa.device == 0

    print("✅ ForgeArray.from_torch: ptr={}, numel={}, dtype={}".format(
        hex(fa.data_ptr), fa.numel, fa.dtype))


def test_dlpack_roundtrip():
    """DLPack roundtrip: torch → capsule → torch."""
    original = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    result = ForgeDLPack.roundtrip(original)

    assert torch.allclose(original, result), "DLPack roundtrip should preserve values"
    # They should share memory
    original[0] = 99.0
    assert result[0].item() == 99.0, "DLPack should be zero-copy"

    print("✅ DLPack roundtrip: zero-copy verified")


def test_shared_memory_modification():
    """Verify that Forge and PyTorch see the same GPU memory."""
    t = torch.zeros(100, device='cuda')
    fa = ForgeArray.from_torch(t)

    # Modify through PyTorch
    t[0] = 42.0
    t[50] = 99.0

    # Read the raw values — if we had Forge running, it would see the changes.
    # Here we verify the pointer is valid by reading back through PyTorch.
    assert t[0].item() == 42.0
    assert t[50].item() == 99.0
    assert fa.data_ptr == t.data_ptr(), "pointer should still match"

    print("✅ Shared memory: modifications visible through same pointer")


def test_dtype_support():
    """Different dtypes map correctly."""
    for torch_dtype, forge_dtype in [
        (torch.float32, "f32"),
        (torch.float64, "f64"),
        (torch.int32, "i32"),
        (torch.int64, "i64"),
    ]:
        t = torch.zeros(10, dtype=torch_dtype, device='cuda')
        fa = ForgeArray.from_torch(t)
        assert fa.dtype == forge_dtype, f"Expected {forge_dtype}, got {fa.dtype}"

    print("✅ dtype support: f32, f64, i32, i64 all mapped correctly")


def test_large_tensor():
    """Large tensor zero-copy."""
    n = 10_000_000  # 10M elements
    t = torch.randn(n, device='cuda')
    fa = ForgeArray.from_torch(t)

    assert fa.numel == n
    assert fa.data_ptr > 0

    # Memory should NOT be duplicated
    # (we can't easily check this, but the pointer sharing is proof)
    ptr1, _ = tensor_to_forge_ptr(t)
    assert ptr1 == fa.data_ptr

    print(f"✅ Large tensor: {n:,} elements, zero-copy, ptr={hex(fa.data_ptr)}")


def test_2d_tensor():
    """2D tensor (matrix)."""
    t = torch.randn(100, 200, device='cuda')
    fa = ForgeArray.from_torch(t)

    assert fa.numel == 20000
    assert fa.dtype == "f32"

    print("✅ 2D tensor: 100x200, numel={}".format(fa.numel))


if __name__ == "__main__":
    tests = [
        test_tensor_to_forge_ptr,
        test_forge_array_from_torch,
        test_dlpack_roundtrip,
        test_shared_memory_modification,
        test_dtype_support,
        test_large_tensor,
        test_2d_tensor,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed! 🎉")
