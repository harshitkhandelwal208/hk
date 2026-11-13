"""
HK's native tensor and autograd core (plan section 1.6 and 161).

Zero dependencies: works entirely through Python's standard library (`ctypes`, `struct`, `math`)
and connects directly to the native HK runtime library (`libhk`) via the C ABI.
Optional bridges allow zero-copy tensor sharing via the Python buffer protocol
(__array_interface__) with NumPy and PyTorch when those packages are present.
"""

from __future__ import annotations

import ctypes
import math
import struct
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# Global autograd state
_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables gradient calculation."""

    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev


class DType:
    def __init__(self, name: str, itemsize: int, format_char: str):
        self.name = name
        self.itemsize = itemsize
        self.format_char = format_char

    def __repr__(self) -> str:
        return f"hk.{self.name}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        return False


float32 = DType("float32", 4, "f")
float16 = DType("float16", 2, "e")
bfloat16 = DType("bfloat16", 2, "H")
float64 = DType("float64", 8, "d")
int8 = DType("int8", 1, "b")
int16 = DType("int16", 2, "h")
int32 = DType("int32", 4, "i")
int64 = DType("int64", 8, "q")
uint8 = DType("uint8", 1, "B")
bool_ = DType("bool_", 1, "?")

f32 = float32
f16 = float16
bf16 = bfloat16
f64 = float64
i8 = int8
i16 = int16
i32 = int32
i64 = int64
u8 = uint8


def _get_native_lib():
    try:
        from . import native

        if native.is_native_available():
            return native.lib
    except Exception:
        pass
    return None


class GradFn:
    """Node in the autograd reverse graph."""

    def __init__(self, name: str, *inputs: "Tensor"):
        self.name = name
        self.inputs = list(inputs)

    def apply(self, grad: "Tensor") -> None:
        pass


class AddBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a, b = self.inputs
        if a.requires_grad:
            g_a = _accumulate_grad(grad, a.shape)
            a._add_grad(g_a)
        if b.requires_grad:
            g_b = _accumulate_grad(grad, b.shape)
            b._add_grad(g_b)


class SubBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a, b = self.inputs
        if a.requires_grad:
            g_a = _accumulate_grad(grad, a.shape)
            a._add_grad(g_a)
        if b.requires_grad:
            g_b = _accumulate_grad(-grad, b.shape)
            b._add_grad(g_b)


class MulBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a, b = self.inputs
        if a.requires_grad:
            g_a = _accumulate_grad(grad * b, a.shape)
            a._add_grad(g_a)
        if b.requires_grad:
            g_b = _accumulate_grad(grad * a, b.shape)
            b._add_grad(g_b)


class DivBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a, b = self.inputs
        if a.requires_grad:
            g_a = _accumulate_grad(grad / b, a.shape)
            a._add_grad(g_a)
        if b.requires_grad:
            g_b = _accumulate_grad(-grad * a / (b * b), b.shape)
            b._add_grad(g_b)


class NegBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a = self.inputs[0]
        if a.requires_grad:
            a._add_grad(-grad)


class MatMulBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a, b = self.inputs
        if a.requires_grad:
            # d(A @ B)/dA = grad @ B^T
            b_t = b.t()
            g_a = grad @ b_t
            a._add_grad(g_a)
        if b.requires_grad:
            # d(A @ B)/dB = A^T @ grad
            a_t = a.t()
            g_b = a_t @ grad
            b._add_grad(g_b)


class SumBackward(GradFn):
    def __init__(self, in_tensor: "Tensor", axis: Optional[int], keepdim: bool):
        super().__init__("SumBackward", in_tensor)
        self.axis = axis
        self.keepdim = keepdim

    def apply(self, grad: "Tensor") -> None:
        inp = self.inputs[0]
        if not inp.requires_grad:
            return
        if self.axis is None:
            g = zeros(inp.shape, dtype=inp.dtype)
            val = grad.item() if grad.numel == 1 else grad.to_list()[0]
            g_data = [val] * inp.numel
            g = tensor(g_data, dtype=inp.dtype).reshape(*inp.shape)
            inp._add_grad(g)
        else:
            # Broadcast grad back along reduced axis
            shape = list(grad.shape)
            if not self.keepdim:
                shape.insert(self.axis, 1)
            g = grad.reshape(*shape)
            # Expand to input shape
            expanded = zeros(inp.shape, dtype=inp.dtype)
            inp_list = inp.to_list()
            g_expanded = _expand_broadcast(g, inp.shape)
            inp._add_grad(g_expanded)


class ReshapeBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a = self.inputs[0]
        if a.requires_grad:
            a._add_grad(grad.reshape(*a.shape))


class ReluBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a = self.inputs[0]
        if a.requires_grad:
            mask = a > 0.0
            a._add_grad(grad * mask)


class SiluBackward(GradFn):
    def apply(self, grad: "Tensor") -> None:
        a = self.inputs[0]
        if a.requires_grad:
            # silu(x) = x * sig(x)
            # silu'(x) = sig(x) * (1 + x * (1 - sig(x)))
            a_data = a.to_list()
            g_data = grad.to_list()
            flat_a = _flatten(a_data) if isinstance(a_data, list) else [a_data]
            flat_g = _flatten(g_data) if isinstance(g_data, list) else [g_data]
            out = []
            for x, g in zip(flat_a, flat_g):
                sig = 1.0 / (1.0 + math.exp(-x)) if -x < 700 else 0.0
                dsilu = sig * (1.0 + x * (1.0 - sig))
                out.append(g * dsilu)
            res = tensor(out, dtype=a.dtype).reshape(*a.shape)
            a._add_grad(res)


def _flatten(seq: Any) -> List[float]:
    if not isinstance(seq, list):
        return [float(seq)]
    res = []
    for item in seq:
        if isinstance(item, list):
            res.extend(_flatten(item))
        else:
            res.append(float(item))
    return res


def _accumulate_grad(grad: "Tensor", target_shape: Tuple[int, ...]) -> "Tensor":
    """Reduces grad to match target_shape if broadcasted."""
    if grad.shape == target_shape:
        return grad
    # Sum out added leading dimensions
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0, keepdim=False)
    # Sum along dimensions that were broadcast from 1
    for dim, (g_d, t_d) in enumerate(zip(grad.shape, target_shape)):
        if t_d == 1 and g_d > 1:
            grad = grad.sum(axis=dim, keepdim=True)
    return grad.reshape(*target_shape)


def _expand_broadcast(t: "Tensor", target_shape: Tuple[int, ...]) -> "Tensor":
    t_flat = _flatten(t.to_list())
    res = zeros(target_shape, dtype=t.dtype)
    # If scalar or 1D broadcast
    if t.numel == 1:
        val = t_flat[0]
        return tensor([val] * res.numel, dtype=t.dtype).reshape(*target_shape)
    return t


class Tensor:
    """A multi-dimensional tensor backed by contiguous storage with autograd support."""

    def __init__(
        self,
        shape: Sequence[int],
        dtype: DType = float32,
        buffer: Optional[bytearray] = None,
        offset: int = 0,
        strides: Optional[Sequence[int]] = None,
        requires_grad: bool = False,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.offset = offset
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self.grad_fn: Optional[GradFn] = None

        # Compute default contiguous strides
        if strides is None:
            self.strides = self._compute_contiguous_strides(self.shape)
        else:
            self.strides = tuple(strides)

        self.numel = 1
        for s in self.shape:
            self.numel *= s

        total_bytes = self.numel * self.dtype.itemsize
        if buffer is None:
            self._buffer = bytearray(total_bytes)
        else:
            self._buffer = buffer

    @staticmethod
    def _compute_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = []
        acc = 1
        for d in reversed(shape):
            strides.append(acc)
            acc *= d
        return tuple(reversed(strides))

    @property
    def data_ptr(self) -> int:
        """Raw memory address of buffer start + offset."""
        c_arr = (ctypes.c_char * len(self._buffer)).from_buffer(self._buffer)
        return ctypes.addressof(c_arr) + self.offset

    @property
    def __array_interface__(self) -> dict:
        """Python buffer interface for zero-copy NumPy/PyTorch sharing."""
        typestr = "<" + self.dtype.format_char
        strides_bytes = tuple(s * self.dtype.itemsize for s in self.strides)
        return {
            "shape": self.shape,
            "typestr": typestr,
            "data": (self.data_ptr, False),
            "version": 3,
            "strides": strides_bytes,
        }

    def _add_grad(self, g: "Tensor") -> None:
        if self.grad is None:
            self.grad = g.detach()
        else:
            self.grad = self.grad + g.detach()

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Computes the sum of gradients of this tensor w.r.t. graph leaves."""
        if not self.requires_grad and self.grad_fn is None:
            raise RuntimeError("Called backward() on a tensor that does not require grad")

        if gradient is None:
            if self.numel != 1:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")
            gradient = ones(self.shape, dtype=self.dtype)

        self._add_grad(gradient)

        # Topological sort of the computational DAG
        topo: List[Tensor] = []
        visited = set()

        def build_topo(t: Tensor):
            if id(t) not in visited:
                visited.add(id(t))
                if t.grad_fn:
                    for inp in t.grad_fn.inputs:
                        build_topo(inp)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            if t.grad_fn is not None and t.grad is not None:
                t.grad_fn.apply(t.grad)

    def detach(self) -> "Tensor":
        """Returns a new Tensor detached from the current graph."""
        res = Tensor(self.shape, self.dtype, bytearray(self._buffer), self.offset, self.strides, requires_grad=False)
        return res

    def item(self) -> float:
        """Returns the value of this tensor as a standard Python number."""
        if self.numel != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        val = struct.unpack_from(self.dtype.format_char, self._buffer, self.offset)[0]
        return float(val)

    def to_list(self) -> Any:
        """Returns the tensor as a nested Python list."""
        flat = []
        fmt = self.dtype.format_char
        size = self.dtype.itemsize
        for i in range(self.numel):
            # Address based on strides
            idx = i
            off = self.offset
            for s, dim in zip(self.strides, self.shape):
                coord = (idx // s) % dim
                off += coord * s * size
            # Contiguous read for fast path
            off_contig = self.offset + i * size
            val = struct.unpack_from(fmt, self._buffer, off_contig)[0]
            flat.append(val)

        def nest(shape, data):
            if not shape:
                return data[0]
            dim = shape[0]
            stride = len(data) // dim
            return [nest(shape[1:], data[j * stride : (j + 1) * stride]) for j in range(dim)]

        return nest(self.shape, flat)

    def reshape(self, *new_shape: int) -> "Tensor":
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])

        new_numel = 1
        infer_dim = -1
        for i, d in enumerate(new_shape):
            if d == -1:
                if infer_dim != -1:
                    raise ValueError("can only specify one unknown dimension")
                infer_dim = i
            else:
                new_numel *= d

        final_shape = list(new_shape)
        if infer_dim != -1:
            final_shape[infer_dim] = self.numel // new_numel
            new_numel *= final_shape[infer_dim]

        if new_numel != self.numel:
            raise ValueError(f"cannot reshape tensor of size {self.numel} into shape {final_shape}")

        res = Tensor(final_shape, self.dtype, self._buffer, self.offset, requires_grad=self.requires_grad)
        if _GRAD_ENABLED and self.requires_grad:
            res.grad_fn = ReshapeBackward("ReshapeBackward", self)
        return res

    def view(self, *shape: int) -> "Tensor":
        return self.reshape(*shape)

    def t(self) -> "Tensor":
        """Transposes a 2D tensor."""
        if len(self.shape) != 2:
            raise RuntimeError("t() expects a 2D tensor")
        return self.transpose(0, 1)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        # Return contiguous copy to keep operations simple
        t_trans = Tensor(new_shape, self.dtype, requires_grad=self.requires_grad)
        flat_self = _flatten(self.to_list())
        # Re-index
        r, c = self.shape
        data = [0.0] * self.numel
        for i in range(r):
            for j in range(c):
                data[j * r + i] = flat_self[i * c + j]
        t_trans = tensor(data, dtype=self.dtype).reshape(*new_shape)
        t_trans.requires_grad = self.requires_grad
        return t_trans

    def __neg__(self) -> "Tensor":
        flat = _flatten(self.to_list())
        out = [-v for v in flat]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and self.requires_grad:
            res.requires_grad = True
            res.grad_fn = NegBackward("NegBackward", self)
        return res

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = tensor([float(other)] * self.numel, dtype=self.dtype).reshape(*self.shape)

        a_data = _flatten(self.to_list())
        b_data = _flatten(other.to_list())
        # Fast elementwise addition
        out = [x + y for x, y in zip(a_data, b_data)]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and (self.requires_grad or other.requires_grad):
            res.requires_grad = True
            res.grad_fn = AddBackward("AddBackward", self, other)
        return res

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = tensor([float(other)] * self.numel, dtype=self.dtype).reshape(*self.shape)
        a_data = _flatten(self.to_list())
        b_data = _flatten(other.to_list())
        out = [x - y for x, y in zip(a_data, b_data)]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and (self.requires_grad or other.requires_grad):
            res.requires_grad = True
            res.grad_fn = SubBackward("SubBackward", self, other)
        return res

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = tensor([float(other)] * self.numel, dtype=self.dtype).reshape(*self.shape)
        a_data = _flatten(self.to_list())
        b_data = _flatten(other.to_list())
        out = [x * y for x, y in zip(a_data, b_data)]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and (self.requires_grad or other.requires_grad):
            res.requires_grad = True
            res.grad_fn = MulBackward("MulBackward", self, other)
        return res

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = tensor([float(other)] * self.numel, dtype=self.dtype).reshape(*self.shape)
        a_data = _flatten(self.to_list())
        b_data = _flatten(other.to_list())
        out = [x / y if y != 0.0 else 0.0 for x, y in zip(a_data, b_data)]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and (self.requires_grad or other.requires_grad):
            res.requires_grad = True
            res.grad_fn = DivBackward("DivBackward", self, other)
        return res

    def __gt__(self, scalar: float) -> "Tensor":
        a_data = _flatten(self.to_list())
        out = [1.0 if v > scalar else 0.0 for v in a_data]
        return tensor(out, dtype=self.dtype).reshape(*self.shape)

    def relu(self) -> "Tensor":
        a_data = _flatten(self.to_list())
        out = [max(0.0, v) for v in a_data]
        res = tensor(out, dtype=self.dtype).reshape(*self.shape)
        if _GRAD_ENABLED and self.requires_grad:
            res.requires_grad = True
            res.grad_fn = ReluBackward("ReluBackward", self)
        return res

    def silu(self) -> "Tensor":
        lib = _get_native_lib()
        if lib is not None and self.dtype == float32 and hasattr(lib, "hk_forward_silu"):
            res = zeros(self.shape, dtype=float32)
            lib.hk_forward_silu(
                (ctypes.c_float * self.numel).from_buffer(self._buffer, self.offset),
                (ctypes.c_float * self.numel).from_buffer(res._buffer),
                ctypes.c_uint64(self.numel),
            )
        else:
            a_data = _flatten(self.to_list())
            out = []
            for v in a_data:
                sig = 1.0 / (1.0 + math.exp(-v)) if -v < 700 else 0.0
                out.append(v * sig)
            res = tensor(out, dtype=self.dtype).reshape(*self.shape)

        if _GRAD_ENABLED and self.requires_grad:
            res.requires_grad = True
            res.grad_fn = SiluBackward("SiluBackward", self)
        return res

    def sum(self, axis: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        a_data = _flatten(self.to_list())
        if axis is None:
            s = sum(a_data)
            res = tensor([s], dtype=self.dtype)
            if not keepdim:
                res = res.reshape()
        else:
            # Axis sum for 2D
            if len(self.shape) == 2:
                r, c = self.shape
                if axis == 0:
                    out = [0.0] * c
                    for i in range(r):
                        for j in range(c):
                            out[j] += a_data[i * c + j]
                    shape = (1, c) if keepdim else (c,)
                    res = tensor(out, dtype=self.dtype).reshape(*shape)
                else:
                    out = [0.0] * r
                    for i in range(r):
                        for j in range(c):
                            out[i] += a_data[i * c + j]
                    shape = (r, 1) if keepdim else (r,)
                    res = tensor(out, dtype=self.dtype).reshape(*shape)
            else:
                s = sum(a_data)
                res = tensor([s], dtype=self.dtype)

        if _GRAD_ENABLED and self.requires_grad:
            res.requires_grad = True
            res.grad_fn = SumBackward(self, axis, keepdim)
        return res

    def mean(self, axis: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        s = self.sum(axis=axis, keepdim=keepdim)
        n = float(self.shape[axis]) if axis is not None else float(self.numel)
        return s / n

    def matmul(self, other: "Tensor") -> "Tensor":
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise NotImplementedError("matmul is currently implemented for 2D matrices")
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError(f"Incompatible shapes for matmul: ({m}, {k}) and ({k2}, {n})")

        res = zeros((m, n), dtype=self.dtype)
        lib = _get_native_lib()

        # Connect directly to native C ABI kernel: hk_gemm_f32
        if lib is not None and self.dtype == float32 and other.dtype == float32 and hasattr(lib, "hk_gemm_f32"):
            lib.hk_gemm_f32(
                (ctypes.c_float * self.numel).from_buffer(self._buffer, self.offset),
                (ctypes.c_float * other.numel).from_buffer(other._buffer, other.offset),
                (ctypes.c_float * (m * n)).from_buffer(res._buffer),
                ctypes.c_uint64(m),
                ctypes.c_uint64(n),
                ctypes.c_uint64(k),
            )
        else:
            # Pure Python reference path
            a_flat = _flatten(self.to_list())
            b_flat = _flatten(other.to_list())
            out = [0.0] * (m * n)
            for i in range(m):
                for j in range(n):
                    s = 0.0
                    for p in range(k):
                        s += a_flat[i * k + p] * b_flat[p * n + j]
                    out[i * n + j] = s
            res = tensor(out, dtype=self.dtype).reshape(m, n)

        if _GRAD_ENABLED and (self.requires_grad or other.requires_grad):
            res.requires_grad = True
            res.grad_fn = MatMulBackward("MatMulBackward", self, other)
        return res

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Tensor":
        """Dot product of two 1D vectors."""
        if self.numel != other.numel:
            raise ValueError("Dot product requires equal number of elements")
        lib = _get_native_lib()
        if lib is not None and self.dtype == float32 and hasattr(lib, "hk_dot_product_f32"):
            val = lib.hk_dot_product_f32(
                (ctypes.c_float * self.numel).from_buffer(self._buffer, self.offset),
                (ctypes.c_float * other.numel).from_buffer(other._buffer, other.offset),
                ctypes.c_uint64(self.numel),
            )
            return tensor([val], dtype=float32)
        return (self * other).sum()

    def rmsnorm(self, weight: "Tensor", eps: float = 1e-6) -> "Tensor":
        """RMSNorm layer using native kernel if available."""
        if len(self.shape) != 2:
            raise NotImplementedError("RMSNorm implemented for 2D (rows, cols)")
        rows, cols = self.shape
        res = zeros(self.shape, dtype=self.dtype)
        lib = _get_native_lib()
        if lib is not None and self.dtype == float32 and hasattr(lib, "hk_forward_rmsnorm"):
            for r in range(rows):
                in_off = self.offset + r * cols * 4
                out_off = r * cols * 4
                lib.hk_forward_rmsnorm(
                    (ctypes.c_float * cols).from_buffer(self._buffer, in_off),
                    (ctypes.c_float * cols).from_buffer(weight._buffer, weight.offset),
                    ctypes.c_float(eps),
                    (ctypes.c_float * cols).from_buffer(res._buffer, out_off),
                    ctypes.c_uint64(cols),
                )
        else:
            # Pure Python implementation
            a_data = self.to_list()
            w_data = _flatten(weight.to_list())
            out = []
            for row in a_data:
                mean_sq = sum(x * x for x in row) / cols
                scale = 1.0 / math.sqrt(mean_sq + eps)
                out.extend([x * scale * w for x, w in zip(row, w_data)])
            res = tensor(out, dtype=self.dtype).reshape(rows, cols)
        return res

    def __repr__(self) -> str:
        data_str = str(self.to_list())
        grad_str = f", requires_grad=True" if self.requires_grad else ""
        return f"hk.Tensor({data_str}, shape={self.shape}, dtype={self.dtype}{grad_str})"


def tensor(data: Any, dtype: Optional[DType] = None, requires_grad: bool = False) -> Tensor:
    """Creates an HK Tensor from nested Python lists or numbers."""
    if isinstance(data, Tensor):
        res = Tensor(data.shape, data.dtype, bytearray(data._buffer), data.offset, requires_grad=requires_grad)
        return res

    # Infer shape
    shape = []
    curr = data
    while isinstance(curr, (list, tuple)):
        shape.append(len(curr))
        curr = curr[0] if len(curr) > 0 else []

    flat = _flatten(data) if shape else [float(data)]
    if not shape:
        shape = [1]

    dt = dtype or float32
    buf = bytearray(len(flat) * dt.itemsize)
    fmt = dt.format_char
    for i, val in enumerate(flat):
        struct.pack_into(fmt, buf, i * dt.itemsize, val)

    return Tensor(shape, dt, buf, requires_grad=requires_grad)


def zeros(shape: Sequence[int], dtype: DType = float32, requires_grad: bool = False) -> Tensor:
    """Returns a tensor filled with the scalar value 0."""
    return Tensor(shape, dtype=dtype, requires_grad=requires_grad)


def ones(shape: Sequence[int], dtype: DType = float32, requires_grad: bool = False) -> Tensor:
    """Returns a tensor filled with the scalar value 1."""
    numel = 1
    for s in shape:
        numel *= s
    return tensor([1.0] * numel, dtype=dtype, requires_grad=requires_grad).reshape(*shape)


def from_buffer(buffer: Any, shape: Sequence[int], dtype: DType = float32, offset: int = 0) -> Tensor:
    """Zero-copy wrap of an external bytes/bytearray/mmap buffer."""
    return Tensor(shape, dtype=dtype, buffer=buffer, offset=offset)


def gradcheck(func: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float = 1e-3, atol: float = 1e-2) -> bool:
    """
    Finite differences gradient checker per plan section 161:
    'Every op's backward is checked against finite differences (a gradcheck).'
    """
    # Run analytical backward
    for inp in inputs:
        inp.grad = None
        inp.requires_grad = True

    out = func(*inputs)
    loss = out.sum()
    loss.backward()

    for inp_idx, inp in enumerate(inputs):
        analytical_grad = _flatten(inp.grad.to_list())
        flat_data = _flatten(inp.to_list())

        for i in range(len(flat_data)):
            # Numerical grad using central differences: (f(x+eps) - f(x-eps)) / (2*eps)
            orig = flat_data[i]

            flat_data[i] = orig + eps
            t_pos = tensor(flat_data, dtype=inp.dtype).reshape(*inp.shape)
            inputs_pos = list(inputs)
            inputs_pos[inp_idx] = t_pos
            out_pos = func(*inputs_pos).sum().item()

            flat_data[i] = orig - eps
            t_neg = tensor(flat_data, dtype=inp.dtype).reshape(*inp.shape)
            inputs_neg = list(inputs)
            inputs_neg[inp_idx] = t_neg
            out_neg = func(*inputs_neg).sum().item()

            flat_data[i] = orig  # restore
            numerical_grad = (out_pos - out_neg) / (2.0 * eps)
            diff = abs(analytical_grad[i] - numerical_grad)
            if diff > atol:
                raise AssertionError(
                    f"Gradcheck failed for input {inp_idx} at index {i}: "
                    f"analytical={analytical_grad[i]:.6f}, numerical={numerical_grad:.6f}, diff={diff:.6f}"
                )
    return True
