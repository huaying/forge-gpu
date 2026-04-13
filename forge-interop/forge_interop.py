"""
forge_interop — Zero-copy bridge between PyTorch tensors and Forge GPU arrays.

Uses DLPack protocol for zero-copy tensor sharing:
- PyTorch tensor → Forge Array (raw GPU pointer)
- Forge Array result → PyTorch tensor

Also provides a ctypes-based wrapper to call Forge compiled kernels from Python.

Usage:
    import torch
    from forge_interop import ForgeKernel, tensor_to_forge_ptr, forge_ptr_to_tensor

    # Share GPU memory
    t = torch.randn(1000, device='cuda')
    ptr, size = tensor_to_forge_ptr(t)  # raw CUDA device pointer

    # Load and run a Forge kernel
    kernel = ForgeKernel("path/to/libforge_example.so")
    kernel.call("my_kernel", args=[ptr, size])
"""

import ctypes
import torch
from typing import Optional, Tuple


def tensor_to_forge_ptr(tensor: torch.Tensor) -> Tuple[int, int]:
    """Extract raw CUDA device pointer from a PyTorch tensor.

    Returns (data_ptr, num_elements).
    The tensor must be contiguous and on a CUDA device.

    This is zero-copy — Forge can read/write through this pointer directly.
    The PyTorch tensor must stay alive while Forge uses the pointer.
    """
    assert tensor.is_cuda, "Tensor must be on CUDA device"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    return tensor.data_ptr(), tensor.numel()


def forge_ptr_to_tensor(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: int = 0,
) -> torch.Tensor:
    """Wrap a raw CUDA device pointer as a PyTorch tensor (zero-copy).

    The caller must ensure the memory stays valid for the tensor's lifetime.

    Uses torch.from_dlpack with a custom DLManagedTensor.
    For simplicity, this creates a tensor from the storage directly.
    """
    # Calculate size
    numel = 1
    for s in shape:
        numel *= s

    # Determine element size
    elem_size = {
        torch.float32: 4,
        torch.float64: 8,
        torch.int32: 4,
        torch.int64: 8,
    }.get(dtype, 4)

    nbytes = numel * elem_size

    # Create an untyped storage from the pointer
    storage = torch.cuda.FloatStorage._new_shared_cuda(device, ptr, nbytes // 4)
    tensor = torch.tensor([], dtype=dtype, device=f'cuda:{device}')
    tensor.set_(storage, 0, shape)
    return tensor


class ForgeDLPack:
    """DLPack-based zero-copy sharing between PyTorch and Forge.

    This is the preferred approach — it's the standard protocol.
    """

    @staticmethod
    def torch_to_capsule(tensor: torch.Tensor):
        """Convert PyTorch tensor to DLPack capsule.

        Pass this capsule to Forge's DLPack import function.
        """
        return torch.utils.dlpack.to_dlpack(tensor)

    @staticmethod
    def capsule_to_torch(capsule) -> torch.Tensor:
        """Convert DLPack capsule (from Forge) to PyTorch tensor."""
        return torch.utils.dlpack.from_dlpack(capsule)

    @staticmethod
    def roundtrip(tensor: torch.Tensor) -> torch.Tensor:
        """Verify DLPack roundtrip works (for testing)."""
        capsule = torch.utils.dlpack.to_dlpack(tensor)
        return torch.utils.dlpack.from_dlpack(capsule)


class ForgeArray:
    """Python wrapper for a Forge GPU array.

    Holds a raw CUDA pointer and metadata, enabling zero-copy
    sharing with PyTorch tensors.

    Supports:
    - __cuda_array_interface__ (CuPy, PyTorch, Numba interop)
    - DLPack protocol (torch.from_dlpack / to_dlpack)
    """

    def __init__(self, data_ptr: int, numel: int, shape: Optional[Tuple[int, ...]] = None,
                 dtype: str = "f32", device: int = 0):
        self.data_ptr = data_ptr
        self.numel = numel
        self.shape_tuple = shape or (numel,)
        self.dtype = dtype
        self.device = device

    @property
    def __cuda_array_interface__(self):
        """Standard CUDA array interface for zero-copy interop.

        Supported by PyTorch, CuPy, Numba, and other GPU frameworks.
        See: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        """
        typestr_map = {
            "f32": "<f4",
            "f64": "<f8",
            "i32": "<i4",
            "i64": "<i8",
            "u32": "<u4",
            "u64": "<u8",
            "f16": "<f2",
        }
        typestr = typestr_map.get(self.dtype, "<f4")

        return {
            "version": 3,
            "shape": self.shape_tuple,
            "typestr": typestr,
            "data": (self.data_ptr, False),  # (ptr, read_only)
            "strides": None,  # C-contiguous
            "stream": None,   # default stream
        }

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "ForgeArray":
        """Create a ForgeArray view of a PyTorch tensor (zero-copy)."""
        assert tensor.is_cuda, "Tensor must be on CUDA"
        assert tensor.is_contiguous(), "Tensor must be contiguous"

        dtype_map = {
            torch.float32: "f32",
            torch.float64: "f64",
            torch.int32: "i32",
            torch.int64: "i64",
            torch.float16: "f16",
        }
        dtype = dtype_map.get(tensor.dtype, "f32")
        device = tensor.device.index or 0

        return cls(
            data_ptr=tensor.data_ptr(),
            numel=tensor.numel(),
            shape=tuple(tensor.shape),
            dtype=dtype,
            device=device,
        )

    def to_torch(self, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Create a PyTorch tensor view of this ForgeArray (zero-copy).

        Uses __cuda_array_interface__ for standard interop.
        The ForgeArray (and underlying allocation) must stay alive.
        """
        if shape is not None:
            self.shape_tuple = shape
        # PyTorch can consume __cuda_array_interface__ directly
        return torch.as_tensor(self, device=f'cuda:{self.device}')


class ForgeKernel:
    """Load and call a compiled Forge shared library from Python.

    This wraps a .so file compiled from Forge Rust code.
    The .so must export C-compatible functions.
    """

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)

    def call(self, func_name: str, args: list):
        """Call a function from the shared library.

        Args should be ctypes-compatible (int, float, ctypes.c_void_p, etc.)
        """
        func = getattr(self.lib, func_name)
        return func(*args)
