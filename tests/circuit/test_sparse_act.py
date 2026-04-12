"""
Unit tests for SparseAct — arithmetic operators, None propagation, to_tensor.
All expected values are hand-computed.
"""
import pytest
import torch
from circuit.types.sparse_act import SparseAct


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

class TestSparseActMul:

    def test_mul_two_sparse_acts_all_fields(self):
        a = SparseAct(
            act=torch.tensor([2.0, 3.0]),
            res=torch.tensor([1.0, -1.0]),
            resc=torch.tensor([4.0]),
        )
        b = SparseAct(
            act=torch.tensor([0.5, 2.0]),
            res=torch.tensor([3.0, 0.0]),
            resc=torch.tensor([-2.0]),
        )
        result = a * b
        torch.testing.assert_close(result.act, torch.tensor([1.0, 6.0]))
        torch.testing.assert_close(result.res, torch.tensor([3.0, 0.0]))
        torch.testing.assert_close(result.resc, torch.tensor([-8.0]))

    def test_mul_none_act_propagates_none(self):
        a = SparseAct(act=None, res=torch.tensor([1.0]))
        b = SparseAct(act=torch.tensor([2.0]), res=torch.tensor([3.0]))
        result = a * b
        assert result.act is None
        torch.testing.assert_close(result.res, torch.tensor([3.0]))

    def test_mul_none_res_propagates_none(self):
        a = SparseAct(act=torch.tensor([2.0]), res=None)
        b = SparseAct(act=torch.tensor([3.0]), res=torch.tensor([1.0]))
        result = a * b
        torch.testing.assert_close(result.act, torch.tensor([6.0]))
        assert result.res is None

    def test_mul_both_none_act(self):
        a = SparseAct(act=None, res=torch.tensor([2.0]))
        b = SparseAct(act=None, res=torch.tensor([3.0]))
        result = a * b
        assert result.act is None
        torch.testing.assert_close(result.res, torch.tensor([6.0]))

    def test_mul_scalar(self):
        a = SparseAct(act=torch.tensor([2.0, 3.0]), res=torch.tensor([1.0]))
        result = a * 3.0
        torch.testing.assert_close(result.act, torch.tensor([6.0, 9.0]))
        torch.testing.assert_close(result.res, torch.tensor([3.0]))

    def test_rmul_scalar(self):
        a = SparseAct(act=torch.tensor([2.0, 3.0]))
        result = 0.5 * a
        torch.testing.assert_close(result.act, torch.tensor([1.0, 1.5]))

    def test_mul_none_act_scalar(self):
        a = SparseAct(act=None, res=torch.tensor([4.0]))
        result = a * 2.0
        assert result.act is None
        torch.testing.assert_close(result.res, torch.tensor([8.0]))


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

class TestSparseActAdd:

    def test_add_two_sparse_acts(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]), res=torch.tensor([0.5]))
        b = SparseAct(act=torch.tensor([3.0, -1.0]), res=torch.tensor([0.5]))
        result = a + b
        torch.testing.assert_close(result.act, torch.tensor([4.0, 1.0]))
        torch.testing.assert_close(result.res, torch.tensor([1.0]))

    def test_add_one_none_act_keeps_other(self):
        a = SparseAct(act=None, res=torch.tensor([1.0]))
        b = SparseAct(act=torch.tensor([2.0]), res=torch.tensor([3.0]))
        result = a + b
        torch.testing.assert_close(result.act, torch.tensor([2.0]))
        torch.testing.assert_close(result.res, torch.tensor([4.0]))

    def test_add_scalar(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]))
        result = a + 10.0
        torch.testing.assert_close(result.act, torch.tensor([11.0, 12.0]))

    def test_radd_scalar(self):
        a = SparseAct(act=torch.tensor([1.0]))
        result = 5.0 + a
        torch.testing.assert_close(result.act, torch.tensor([6.0]))


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------

class TestSparseActSub:

    def test_sub_two_sparse_acts(self):
        a = SparseAct(act=torch.tensor([5.0, 3.0]))
        b = SparseAct(act=torch.tensor([1.0, 4.0]))
        result = a - b
        torch.testing.assert_close(result.act, torch.tensor([4.0, -1.0]))

    def test_sub_none_first_negates_second(self):
        a = SparseAct(act=None)
        b = SparseAct(act=torch.tensor([2.0]))
        result = a - b
        torch.testing.assert_close(result.act, torch.tensor([-2.0]))

    def test_sub_scalar(self):
        a = SparseAct(act=torch.tensor([5.0, 3.0]))
        result = a - 1.0
        torch.testing.assert_close(result.act, torch.tensor([4.0, 2.0]))


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

class TestSparseActNeg:

    def test_neg(self):
        a = SparseAct(act=torch.tensor([1.0, -2.0]), res=torch.tensor([3.0]))
        result = -a
        torch.testing.assert_close(result.act, torch.tensor([-1.0, 2.0]))
        torch.testing.assert_close(result.res, torch.tensor([-3.0]))


# ---------------------------------------------------------------------------
# Matmul (dot-product)
# ---------------------------------------------------------------------------

class TestSparseActMatmul:

    def test_matmul_element_wise_act_contracted_res(self):
        """
        @ operator: act element-wise, res dot-product contracted to resc.
        a.act=[2,4], b.act=[3,5] → act=[6,20]
        a.res=[1,2], b.res=[0.5,0.5] → resc = sum([0.5, 1.0]) = [1.5]
        """
        a = SparseAct(act=torch.tensor([2.0, 4.0]), res=torch.tensor([1.0, 2.0]))
        b = SparseAct(act=torch.tensor([3.0, 5.0]), res=torch.tensor([0.5, 0.5]))
        result = a @ b
        torch.testing.assert_close(result.act, torch.tensor([6.0, 20.0]))
        torch.testing.assert_close(result.resc, torch.tensor([1.5]))
        assert result.res is None

    def test_matmul_none_act(self):
        a = SparseAct(act=None, res=torch.tensor([1.0, 2.0]))
        b = SparseAct(act=None, res=torch.tensor([3.0, 4.0]))
        result = a @ b
        assert result.act is None
        torch.testing.assert_close(result.resc, torch.tensor([11.0]))


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------

class TestSparseActDiv:

    def test_div_scalar(self):
        a = SparseAct(act=torch.tensor([6.0, 9.0]), res=torch.tensor([4.0]))
        result = a / 3.0
        torch.testing.assert_close(result.act, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(result.res, torch.tensor([4.0 / 3.0]))

    def test_div_sparse_act(self):
        a = SparseAct(act=torch.tensor([6.0, 8.0]))
        b = SparseAct(act=torch.tensor([2.0, 4.0]))
        result = a / b
        torch.testing.assert_close(result.act, torch.tensor([3.0, 2.0]))


# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------

class TestSparseActToTensor:

    def test_act_and_res_concatenated(self):
        """res is contracted (summed) to 1D then concatenated with act."""
        a = SparseAct(act=torch.tensor([1.0, 2.0]), res=torch.tensor([3.0, 4.0]))
        result = a.to_tensor()
        # res.sum(dim=-1, keepdim=True) = [7.0]
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 7.0]))

    def test_act_and_resc_concatenated(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]), resc=torch.tensor([5.0]))
        result = a.to_tensor()
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 5.0]))

    def test_act_only(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]))
        result = a.to_tensor()
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))

    def test_res_only_contracts(self):
        a = SparseAct(res=torch.tensor([3.0, 4.0]))
        result = a.to_tensor()
        torch.testing.assert_close(result, torch.tensor([7.0]))

    def test_all_none_returns_empty(self):
        a = SparseAct()
        result = a.to_tensor()
        assert result.numel() == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestSparseActProperties:

    def test_shape_delegates_to_act(self):
        a = SparseAct(act=torch.zeros(2, 3, 4))
        assert a.shape == torch.Size([2, 3, 4])

    def test_shape_empty_when_no_act(self):
        a = SparseAct(res=torch.zeros(2, 3))
        assert a.shape == torch.Size([])

    def test_device_from_act(self):
        a = SparseAct(act=torch.zeros(2))
        assert a.device == torch.device("cpu")

    def test_device_fallback_to_res(self):
        a = SparseAct(res=torch.zeros(2))
        assert a.device == torch.device("cpu")

    def test_device_fallback_to_resc(self):
        a = SparseAct(resc=torch.zeros(1))
        assert a.device == torch.device("cpu")

    def test_requires_grad_true(self):
        a = SparseAct(act=torch.zeros(2, requires_grad=True))
        assert a.requires_grad is True

    def test_requires_grad_false_no_act(self):
        a = SparseAct(res=torch.zeros(2))
        assert a.requires_grad is False

    def test_grad_fn_present(self):
        x = torch.tensor([1.0], requires_grad=True)
        a = SparseAct(act=x * 2)
        assert a.grad_fn is not None

    def test_grad_fn_none_for_leaf(self):
        a = SparseAct(act=torch.tensor([1.0]))
        assert a.grad_fn is None


# ---------------------------------------------------------------------------
# sum / mean
# ---------------------------------------------------------------------------

class TestSparseActReductions:

    def test_sum_no_dim(self):
        a = SparseAct(act=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = a.sum()
        torch.testing.assert_close(result.act, torch.tensor(10.0))

    def test_sum_with_dim(self):
        a = SparseAct(act=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = a.sum(dim=0)
        torch.testing.assert_close(result.act, torch.tensor([4.0, 6.0]))

    def test_mean_no_dim(self):
        a = SparseAct(act=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = a.mean()
        torch.testing.assert_close(result.act, torch.tensor(2.5))

    def test_mean_with_dim(self):
        a = SparseAct(act=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = a.mean(dim=1)
        torch.testing.assert_close(result.act, torch.tensor([1.5, 3.5]))


# ---------------------------------------------------------------------------
# clone / detach
# ---------------------------------------------------------------------------

class TestSparseActCloneDetach:

    def test_clone_is_independent(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]))
        b = a.clone()
        b.act[0] = 99.0
        assert a.act[0].item() == 1.0

    def test_detach_removes_grad_fn(self):
        x = torch.tensor([1.0], requires_grad=True)
        a = SparseAct(act=x * 2)
        assert a.grad_fn is not None
        b = a.detach()
        assert b.grad_fn is None


# ---------------------------------------------------------------------------
# getitem
# ---------------------------------------------------------------------------

class TestSparseActGetitem:

    def test_getitem_delegates_to_act(self):
        a = SparseAct(act=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = a[0]
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))

    def test_getitem_fancy_indexing(self):
        a = SparseAct(act=torch.tensor([10.0, 20.0, 30.0]))
        result = a[torch.tensor([0, 2])]
        torch.testing.assert_close(result, torch.tensor([10.0, 30.0]))


# ---------------------------------------------------------------------------
# abs / zeros_like / ones_like
# ---------------------------------------------------------------------------

class TestSparseActMisc:

    def test_abs(self):
        a = SparseAct(act=torch.tensor([-1.0, 2.0, -3.0]))
        result = a.abs()
        torch.testing.assert_close(result.act, torch.tensor([1.0, 2.0, 3.0]))

    def test_zeros_like(self):
        a = SparseAct(act=torch.tensor([1.0, 2.0]), res=torch.tensor([3.0]))
        result = a.zeros_like()
        torch.testing.assert_close(result.act, torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(result.res, torch.tensor([0.0]))

    def test_ones_like(self):
        a = SparseAct(act=torch.tensor([5.0, 6.0]))
        result = a.ones_like()
        torch.testing.assert_close(result.act, torch.tensor([1.0, 1.0]))

    def test_repr(self):
        a = SparseAct(act=torch.zeros(2, 3), res=torch.zeros(2, 4))
        r = repr(a)
        assert "act_shape" in r
        assert "res_shape" in r

    def test_repr_none_act(self):
        a = SparseAct()
        assert "act=None" in repr(a)
