"""Tests for the TorchScript and ONNX export scripts.

Each exporter is tested by:
1. Building a tiny ``AlphaZeroNetwork`` and saving a checkpoint.
2. Running the export function on it.
3. Loading the exported artifact back and comparing outputs against
   the original PyTorch network on a fresh input.

ONNX round-trip checks require ``onnxruntime``; tests that need it
``pytest.importorskip`` so contributors without the optional dep
still get a green test run for the parts that do not need it.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

# Make the scripts/ directory importable so we can call the export
# functions directly (they are not part of the installed package).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "alphazero"))

from export_onnx import export_onnx  # noqa: E402
from export_torchscript import export_torchscript  # noqa: E402

from shogi_dojo.alphazero.encoding import NUM_INPUT_CHANNELS  # noqa: E402
from shogi_dojo.alphazero.network import AlphaZeroNetwork  # noqa: E402
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS  # noqa: E402


# Tiny architecture that keeps tests fast (~milliseconds per export).
TINY = dict(num_blocks=1, num_filters=8, value_head_hidden=8)


def _save_tiny_checkpoint(path: str) -> AlphaZeroNetwork:
    """Build a tiny network, save it as a checkpoint, return the live module."""
    torch.manual_seed(0)
    net = AlphaZeroNetwork(**TINY)
    net.eval()
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
        },
        path,
    )
    return net


# ---------------------------------------------------------------------------
# TorchScript
# ---------------------------------------------------------------------------


class TestExportTorchScript:
    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.ts")
            _save_tiny_checkpoint(ckpt)
            export_torchscript(ckpt, out, **TINY)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0

    def test_loaded_module_matches_pytorch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.ts")
            net = _save_tiny_checkpoint(ckpt)
            export_torchscript(ckpt, out, **TINY)

            traced = torch.jit.load(out)
            traced.eval()
            x = torch.randn(2, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
            with torch.no_grad():
                ref_policy, ref_value = net(x)
                out_policy, out_value = traced(x)
            assert torch.allclose(ref_policy, out_policy, atol=1e-5)
            assert torch.allclose(ref_value, out_value, atol=1e-5)

    def test_supports_dynamic_batch_size(self):
        # Trace with batch=1; verify the loaded module accepts any batch.
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.ts")
            _save_tiny_checkpoint(ckpt)
            export_torchscript(ckpt, out, **TINY)
            traced = torch.jit.load(out)
            traced.eval()

            for batch in (1, 4, 16):
                x = torch.randn(batch, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
                with torch.no_grad():
                    p, v = traced(x)
                assert p.shape == (batch, 139, NUM_RANKS, NUM_FILES)
                assert v.shape == (batch, 1)


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


class TestExportOnnx:
    # ONNX export needs the optional `onnx` package (install via
    # `pip install -e ".[export]"`). The autouse fixture skips every
    # test in this class when it is missing, so the suite stays green
    # for contributors who do not need to run exports themselves.
    @pytest.fixture(autouse=True)
    def _require_onnx(self):
        pytest.importorskip("onnx")

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.onnx")
            _save_tiny_checkpoint(ckpt)
            export_onnx(ckpt, out, verify=False, **TINY)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0

    def test_round_trip_matches_pytorch(self):
        # The export's own --verify path covers this, but we exercise
        # it explicitly here too — and call onnxruntime ourselves so a
        # missing dependency surfaces as a clear skip rather than a
        # silent log line.
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.onnx")
            net = _save_tiny_checkpoint(ckpt)
            export_onnx(ckpt, out, verify=False, **TINY)

            sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
            x = torch.randn(3, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
            onnx_outs = sess.run(None, {"board": x.numpy()})
            with torch.no_grad():
                ref_policy, ref_value = net(x)
            assert np.allclose(onnx_outs[0], ref_policy.numpy(), atol=1e-4)
            assert np.allclose(onnx_outs[1], ref_value.numpy(), atol=1e-4)

    def test_supports_dynamic_batch_size(self):
        ort = pytest.importorskip("onnxruntime")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.onnx")
            _save_tiny_checkpoint(ckpt)
            export_onnx(ckpt, out, verify=False, **TINY)

            sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
            for batch in (1, 5, 8):
                x = torch.randn(batch, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
                outs = sess.run(None, {"board": x.numpy()})
                assert outs[0].shape == (batch, 139, NUM_RANKS, NUM_FILES)
                assert outs[1].shape == (batch, 1)

    def test_verify_path_is_a_no_op_when_onnxruntime_missing(self, monkeypatch):
        # If onnxruntime is not importable, the verify branch should
        # log and return cleanly rather than raise. Simulate the
        # missing-dep case by hiding the module.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("simulated missing onnxruntime")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "tiny.pt")
            out = os.path.join(tmpdir, "tiny.onnx")
            _save_tiny_checkpoint(ckpt)
            # verify=True but onnxruntime "missing" → must not raise.
            export_onnx(ckpt, out, verify=True, **TINY)
            assert os.path.exists(out)
