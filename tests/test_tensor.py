"""
Tests for hk.tensor and autograd (plan section 1.6 and 161).
Zero dependencies: runs without torch or numpy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import hk
from hk.tensor import Tensor, tensor, zeros, ones, from_buffer, no_grad, gradcheck, float32


def test_tensor_creation_and_properties():
    t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32)
    assert t.shape == (2, 3)
    assert t.numel == 6
    assert t.dtype == float32
    assert t.to_list() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    z = zeros((3, 2), dtype=float32)
    assert z.shape == (3, 2)
    assert z.to_list() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    o = ones((2, 2), dtype=float32)
    assert o.to_list() == [[1.0, 1.0], [1.0, 1.0]]


def test_tensor_arithmetic():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])

    c = a + b
    assert c.to_list() == [[6.0, 8.0], [10.0, 12.0]]

    d = b - a
    assert d.to_list() == [[4.0, 4.0], [4.0, 4.0]]

    e = a * b
    assert e.to_list() == [[5.0, 12.0], [21.0, 32.0]]

    f = -a
    assert f.to_list() == [[-1.0, -2.0], [-3.0, -4.0]]


def test_tensor_matmul():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[2.0, 0.0], [1.0, 2.0]])
    c = a @ b
    # [[1*2 + 2*1, 1*0 + 2*2], [3*2 + 4*1, 3*0 + 4*2]] = [[4, 4], [10, 8]]
    assert c.to_list() == [[4.0, 4.0], [10.0, 8.0]]


def test_tensor_activations():
    a = tensor([[-1.0, 2.0], [0.0, -3.0]])
    r = a.relu()
    assert r.to_list() == [[0.0, 2.0], [0.0, 0.0]]

    s = tensor([0.0]).silu()
    assert abs(s.item() - 0.0) < 1e-6


def test_autograd_basic():
    x = tensor([[2.0, 3.0]], requires_grad=True)
    y = x * x + x * 3.0
    # dy/dx = 2*x + 3 -> at x=[2, 3], dy/dx = [7, 9]
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 2)
    assert abs(x.grad.to_list()[0][0] - 7.0) < 1e-5
    assert abs(x.grad.to_list()[0][1] - 9.0) < 1e-5


def test_autograd_matmul():
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[2.0, 1.0], [0.0, 3.0]], requires_grad=True)
    c = a @ b
    loss = c.sum()
    loss.backward()

    # d(sum(A @ B))/dA = ones @ B^T = [[2+1, 0+3], [2+1, 0+3]] = [[3, 3], [3, 3]]
    assert a.grad is not None
    assert a.grad.to_list() == [[3.0, 3.0], [3.0, 3.0]]
    # d(sum(A @ B))/dB = A^T @ ones = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
    assert b.grad is not None
    assert b.grad.to_list() == [[4.0, 4.0], [6.0, 6.0]]


def test_gradcheck_operations():
    # Verify gradcheck with finite differences per plan section 161
    def fn_add(x):
        return (x * 2.5 + 1.5).relu()

    x = tensor([[1.5, -0.5], [2.0, 3.0]], requires_grad=True)
    assert gradcheck(fn_add, [x])

    def fn_matmul(x, w):
        return x @ w

    x2 = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w2 = tensor([[0.5, -1.0], [1.5, 2.0]], requires_grad=True)
    assert gradcheck(fn_matmul, [x2, w2])


def test_no_grad():
    x = tensor([2.0], requires_grad=True)
    with no_grad():
        y = x * 2.0
    assert y.requires_grad is False
    assert y.grad_fn is None


def test_buffer_protocol():
    t = tensor([1.0, 2.0, 3.0, 4.0], dtype=float32)
    interface = t.__array_interface__
    assert interface["shape"] == (4,)
    assert interface["typestr"] == "<f"
    assert interface["data"][0] == t.data_ptr


if __name__ == "__main__":
    test_tensor_creation_and_properties()
    test_tensor_arithmetic()
    test_tensor_matmul()
    test_tensor_activations()
    test_autograd_basic()
    test_autograd_matmul()
    test_gradcheck_operations()
    test_no_grad()
    test_buffer_protocol()
    print("All hk.tensor and autograd tests passed successfully!")
