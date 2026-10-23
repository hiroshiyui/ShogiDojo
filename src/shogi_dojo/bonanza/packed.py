"""Pre-processed ("packed") kifu dataset for fast Bonanza training.

Why this module exists
----------------------

The streaming pipeline in :mod:`shogi_dojo.bonanza.dataset` re-does the
same expensive work every epoch: parse each CSA file, replay the game
on a python-shogi board, and — the slowest part — enumerate every legal
move at every position to build the legal-move mask. On the v1 run this
capped training at ~2,200 samples/s while the GPU could do ~27,000/s,
so the GPU sat ~90% idle.

None of that work changes between epochs. This module does it **once**,
writes the results to disk in a compact binary format, and then serves
training batches from memory-mapped NumPy arrays. Decoding a batch is a
handful of vectorized NumPy operations; python-shogi is not involved at
training time at all.

Pipeline::

    CSA files ──(once: preprocess_csa_files)──► packed shards on disk
    packed shards ──(every epoch: PackedDataset)──► training batches

On-disk layout
--------------

A packed dataset is a directory::

    <root>/
      config.json          # format version + replay settings (min/max ply)
      shard_00000/
        meta.json          # num_samples, num_games, failed files, ...
        files.txt          # the CSA files this shard was built from
        planes.npy         # (N, 284)  uint8  — bit-packed piece planes 0..27
        scalars.npy        # (N, 17)   float32 — constant planes 28..44
        target.npy         # (N,)      uint16 — professional's move index
        value.npy          # (N,)      int8   — game outcome for STM (+1/0/-1)
        legal_offsets.npy  # (N + 1,)  int64  — row pointers into legal_indices
        legal_indices.npy  # (M,)      uint16 — flat policy indices of legal moves
      shard_00001/
      ...

How the 45-channel input tensor is compressed (see
:mod:`shogi_dojo.alphazero.encoding` for the channel layout):

- **Channels 0..27** are the own/opponent piece planes. Every value is
  exactly 0.0 or 1.0, so we store them as bits: 28 × 81 = 2,268 bits =
  284 bytes (vs 9 KB as float32).
- **Channels 28..44** (hand counts, repetition, side-to-move, move
  number) each hold a single value broadcast over the whole 9×9 plane.
  We store just that one float per channel: 17 × 4 = 68 bytes.

The legal-move mask has 11,259 entries but only ~80–120 of them are
``True`` in a typical position, so instead of storing the mask we store
the list of ``True`` positions ("CSR" / ragged-array layout):
``legal_indices[legal_offsets[i]:legal_offsets[i + 1]]`` are the legal
move indices of sample ``i``.

The result is ~550 bytes per sample instead of ~26 KB, and the decoded
tensors are **bit-for-bit identical** to what
:func:`shogi_dojo.bonanza.dataset.replay_game` produces (the test suite
checks this).

Each shard is written to a temporary directory and renamed into place
only when complete, so an interrupted preprocessing run can be resumed:
finished shards are skipped, half-written ones are redone.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from shogi_dojo.alphazero.encoding import (
    NUM_INPUT_CHANNELS,
    NUM_MOVE_PLANES,
    POLICY_DIM,
)
from shogi_dojo.bonanza.dataset import parse_csa_file, replay_game
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS, NUM_SQUARES

logger = logging.getLogger(__name__)


# ============================================================================
# Format constants
# ============================================================================

# Bump this whenever the on-disk layout changes. PackedDataset refuses to
# open a directory written with a different version rather than silently
# decoding garbage.
FORMAT_VERSION = 1

# Channels [0, NUM_BINARY_PLANES) are 0/1 piece planes → stored as bits.
NUM_BINARY_PLANES = 28
NUM_PLANE_BITS = NUM_BINARY_PLANES * NUM_SQUARES          # 2,268
PLANE_BYTES = (NUM_PLANE_BITS + 7) // 8                   # 284

# Channels [NUM_BINARY_PLANES, NUM_INPUT_CHANNELS) are constant per plane
# → stored as one float32 each.
NUM_SCALAR_PLANES = NUM_INPUT_CHANNELS - NUM_BINARY_PLANES  # 17

# Policy indices go up to 11,258, which fits comfortably in uint16 (max 65,535).
assert POLICY_DIM <= np.iinfo(np.uint16).max + 1

CONFIG_FILE = "config.json"
SHARD_META_FILE = "meta.json"
SHARD_FILES_LIST = "files.txt"
SHARD_PREFIX = "shard_"
_SHARD_DIR_RE = re.compile(rf"{SHARD_PREFIX}\d{{5}}")

# Every array file inside a shard, and the dtype it must have.
_SHARD_ARRAYS = {
    "planes": np.uint8,
    "scalars": np.float32,
    "target": np.uint16,
    "value": np.int8,
    "legal_offsets": np.int64,
    "legal_indices": np.uint16,
}


# ============================================================================
# Packing: replay_game samples → compact NumPy arrays
# ============================================================================


def pack_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    """Compress a list of :func:`replay_game` samples into shard arrays.

    Args:
        samples: Sample dicts with keys ``"input"`` (45×9×9 float tensor),
            ``"target"`` (int), ``"legal_mask"`` (139×9×9 bool tensor),
            ``"value_target"`` (float in {-1, 0, +1}).

    Returns:
        A dict with one NumPy array per entry of the on-disk layout
        (see the module docstring).

    Raises:
        ValueError: If an input violates the assumptions the compression
            relies on (piece planes not 0/1, or a "scalar" plane that is
            not constant). This guards against someone extending
            ``encode_position`` without updating this format.
    """
    n = len(samples)
    if n == 0:
        return _empty_arrays()

    inputs = torch.stack([s["input"] for s in samples]).numpy()  # (N, 45, 9, 9)

    # --- Piece planes → bits ---
    binary = inputs[:, :NUM_BINARY_PLANES].reshape(n, NUM_PLANE_BITS)
    if not np.all((binary == 0.0) | (binary == 1.0)):
        raise ValueError("Piece planes 0..27 must contain only 0.0 / 1.0.")
    planes = np.packbits(binary.astype(np.uint8), axis=1)  # (N, 284)

    # --- Constant planes → one float each ---
    scalar_part = inputs[:, NUM_BINARY_PLANES:]              # (N, 17, 9, 9)
    scalars = scalar_part[:, :, 0, 0].copy()                 # (N, 17)
    if not np.array_equal(scalar_part, np.broadcast_to(scalars[:, :, None, None], scalar_part.shape)):
        raise ValueError(
            f"Planes {NUM_BINARY_PLANES}..{NUM_INPUT_CHANNELS - 1} must be "
            "constant over the board to be packed as scalars."
        )

    # --- Legal mask → ragged list of flat indices ---
    masks = torch.stack([s["legal_mask"] for s in samples]).reshape(n, POLICY_DIM).numpy()
    rows, cols = np.nonzero(masks)  # rows come out sorted, so cols are grouped per sample
    counts = np.bincount(rows, minlength=n)
    legal_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=legal_offsets[1:])

    values = np.array([s["value_target"] for s in samples], dtype=np.float32)
    if not np.all(np.isin(values, (-1.0, 0.0, 1.0))):
        raise ValueError("value_target must be -1, 0 or +1.")

    return {
        "planes": planes,
        "scalars": scalars.astype(np.float32),
        "target": np.array([s["target"] for s in samples], dtype=np.uint16),
        "value": values.astype(np.int8),
        "legal_offsets": legal_offsets,
        "legal_indices": cols.astype(np.uint16),
    }


def _empty_arrays() -> dict[str, np.ndarray]:
    """Shard arrays for zero samples (keeps shapes consistent)."""
    return {
        "planes": np.zeros((0, PLANE_BYTES), dtype=np.uint8),
        "scalars": np.zeros((0, NUM_SCALAR_PLANES), dtype=np.float32),
        "target": np.zeros(0, dtype=np.uint16),
        "value": np.zeros(0, dtype=np.int8),
        "legal_offsets": np.zeros(1, dtype=np.int64),
        "legal_indices": np.zeros(0, dtype=np.uint16),
    }


def unpack_batch(
    planes: np.ndarray,
    scalars: np.ndarray,
    target: np.ndarray,
    value: np.ndarray,
    legal_lists: list[np.ndarray],
) -> dict[str, torch.Tensor]:
    """Inverse of :func:`pack_samples` for a batch of B samples.

    Args:
        planes: ``(B, 284)`` uint8 bit-packed piece planes.
        scalars: ``(B, 17)`` float32 constant-plane values.
        target: ``(B,)`` move indices.
        value: ``(B,)`` outcomes in {-1, 0, +1}.
        legal_lists: B arrays, each holding that sample's legal move
            indices.

    Returns:
        The same batch dict the trainer expects from the other datasets:
        ``input`` (B, 45, 9, 9) float32, ``target`` (B,) int64,
        ``legal_mask`` (B, 139, 9, 9) bool, ``value_target`` (B,) float32.
    """
    b = planes.shape[0]

    inputs = np.empty((b, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES), dtype=np.float32)
    bits = np.unpackbits(planes, axis=1, count=NUM_PLANE_BITS)
    inputs[:, :NUM_BINARY_PLANES] = bits.reshape(b, NUM_BINARY_PLANES, NUM_RANKS, NUM_FILES)
    inputs[:, NUM_BINARY_PLANES:] = scalars[:, :, None, None]

    # Scatter every sample's legal indices into one (B, 11259) mask in a
    # single fancy-indexing assignment: row_ids repeats each sample's row
    # number once per legal move it has.
    mask = np.zeros((b, POLICY_DIM), dtype=bool)
    if b:
        lengths = np.fromiter((len(x) for x in legal_lists), dtype=np.int64, count=b)
        row_ids = np.repeat(np.arange(b), lengths)
        mask[row_ids, np.concatenate(legal_lists).astype(np.int64)] = True

    return {
        "input": torch.from_numpy(inputs),
        "target": torch.from_numpy(target.astype(np.int64)),
        "legal_mask": torch.from_numpy(mask).view(b, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES),
        "value_target": torch.from_numpy(value.astype(np.float32)),
    }


# ============================================================================
# Writing shards
# ============================================================================


def _shard_name(shard_id: int) -> str:
    return f"{SHARD_PREFIX}{shard_id:05d}"


def _shard_id(shard_dir: Path) -> int:
    """Inverse of :func:`_shard_name`: ``shard_00042`` → ``42``."""
    return int(shard_dir.name[len(SHARD_PREFIX):])


def write_shard(
    out_dir: str | Path,
    shard_id: int,
    arrays: dict[str, np.ndarray],
    source_files: list[str],
    num_games: int,
    failed_files: list[str],
) -> Path:
    """Write one shard atomically and return its final directory.

    The shard is first written to ``shard_XXXXX.tmp`` and then renamed.
    A rename within one filesystem is atomic, so a crash leaves either a
    complete shard or a ``.tmp`` leftover — never a half-written shard
    that looks complete.
    """
    out_dir = Path(out_dir)
    final = out_dir / _shard_name(shard_id)
    tmp = out_dir / (_shard_name(shard_id) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    for name, dtype in _SHARD_ARRAYS.items():
        np.save(tmp / f"{name}.npy", arrays[name].astype(dtype, copy=False))

    (tmp / SHARD_FILES_LIST).write_text("".join(f"{f}\n" for f in source_files))
    meta = {
        "format_version": FORMAT_VERSION,
        "num_samples": int(arrays["target"].shape[0]),
        "num_games": num_games,
        "num_files": len(source_files),
        "failed_files": failed_files,
    }
    (tmp / SHARD_META_FILE).write_text(json.dumps(meta, indent=2))

    os.replace(tmp, final)
    return final


def _build_shard(task: tuple) -> dict:
    """Worker entry point: replay one chunk of CSA files into one shard.

    Runs in a separate process (see :func:`preprocess_csa_files`), so it
    takes a single picklable tuple and returns a small summary dict.
    """
    shard_id, files, out_dir, min_ply, max_ply = task

    samples: list[dict] = []
    num_games = 0
    failed: list[str] = []
    for path in files:
        try:
            games = parse_csa_file(path)
        except Exception as e:  # unreadable file, bad encoding, ...
            logger.debug("Skipping %s: %s", path, e)
            failed.append(path)
            continue
        for game in games:
            samples.extend(replay_game(game, min_ply=min_ply, max_ply=max_ply))
            num_games += 1

    arrays = pack_samples(samples)
    write_shard(out_dir, shard_id, arrays, files, num_games, failed)
    return {
        "shard_id": shard_id,
        "num_samples": int(arrays["target"].shape[0]),
        "num_games": num_games,
        "num_failed": len(failed),
    }


def _init_worker() -> None:
    # Each worker is single-threaded Python; stop PyTorch from also
    # spawning its own intra-op threads in every process, which would
    # oversubscribe the CPU.
    torch.set_num_threads(1)


@dataclass
class PreprocessSummary:
    """What :func:`preprocess_csa_files` did."""

    shards_built: int
    shards_skipped: int
    samples_built: int
    games_built: int
    failed_files: int


def preprocess_csa_files(
    csa_files: list[str],
    out_dir: str | Path,
    *,
    files_per_shard: int = 1000,
    min_ply: int = 1,
    max_ply: int = 512,
    num_workers: int = 1,
    progress: Optional[Callable[[dict], None]] = None,
    max_pool_restarts: int = 3,
) -> PreprocessSummary:
    """Convert CSA files into a packed dataset directory.

    Files are split into consecutive chunks of ``files_per_shard``; each
    chunk becomes one shard, built by one worker process. Re-running with
    the same arguments resumes: shards that already exist are skipped.

    Args:
        csa_files: CSA file paths. Their order decides which shard each
            file lands in, so keep the list stable when resuming.
        out_dir: Output directory (created if missing).
        files_per_shard: CSA files per shard. With wdoor (one game per
            file, ~100 positions per game) the default 1,000 gives
            shards of ~100K samples / ~55 MB.
        min_ply, max_ply: Passed to :func:`replay_game`. Baked into the
            packed data, so they are recorded in ``config.json`` and
            resuming with different values is refused.
        num_workers: Parallel worker processes. ``1`` runs in-process
            (simplest to debug).
        progress: Optional callback, called with each finished shard's
            summary dict.
        max_pool_restarts: With ``num_workers > 1``: how many times to
            restart the worker pool after a worker process dies (e.g.
            segfault or out-of-memory kill), retrying the shards that
            were unfinished. If workers keep dying after that, raises
            ``RuntimeError`` instead of hanging.

    Returns:
        A :class:`PreprocessSummary`.
    """
    if max_pool_restarts < 0:
        raise ValueError(f"max_pool_restarts must be ≥ 0, got {max_pool_restarts}")
    if files_per_shard < 1:
        raise ValueError(f"files_per_shard must be ≥ 1, got {files_per_shard}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "format_version": FORMAT_VERSION,
        "min_ply": min_ply,
        "max_ply": max_ply,
        "files_per_shard": files_per_shard,
        "num_files": len(csa_files),
    }
    config_path = out_dir / CONFIG_FILE
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise ValueError(
                f"{out_dir} was created with different settings "
                f"({existing}); refusing to mix them with {config}. "
                "Use a fresh output directory."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))

    tasks = []
    skipped = 0
    for shard_id, start in enumerate(range(0, len(csa_files), files_per_shard)):
        if (out_dir / _shard_name(shard_id) / SHARD_META_FILE).exists():
            skipped += 1
            continue
        chunk = list(csa_files[start:start + files_per_shard])
        tasks.append((shard_id, chunk, str(out_dir), min_ply, max_ply))

    summary = PreprocessSummary(0, skipped, 0, 0, 0)

    def _record(result: dict) -> None:
        summary.shards_built += 1
        summary.samples_built += result["num_samples"]
        summary.games_built += result["num_games"]
        summary.failed_files += result["num_failed"]
        if progress is not None:
            progress(result)

    if num_workers <= 1:
        for task in tasks:
            _record(_build_shard(task))
    else:
        _run_in_pool(tasks, num_workers, max_pool_restarts, _record)

    return summary


def _finished_shard_summary(task: tuple) -> Optional[dict]:
    """Summary dict for a task whose shard is already complete on disk.

    Returns the same dict ``_build_shard`` would have returned, read back
    from the shard's ``meta.json``, or ``None`` if the shard is not
    finished. A shard directory only appears (atomically renamed from
    ``.tmp``) after ``meta.json`` is written, so its presence means done.
    """
    shard_id, _files, out_dir, _min_ply, _max_ply = task
    meta_path = Path(out_dir) / _shard_name(shard_id) / SHARD_META_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return {
        "shard_id": shard_id,
        "num_samples": meta["num_samples"],
        "num_games": meta["num_games"],
        "num_failed": len(meta["failed_files"]),
    }


def _run_in_pool(
    tasks: list[tuple],
    num_workers: int,
    max_pool_restarts: int,
    record: Callable[[dict], None],
) -> None:
    """Build shards in worker processes, surviving worker crashes.

    Why not ``multiprocessing.Pool``: if a ``Pool`` worker dies abruptly
    (segfault, out-of-memory kill), the pool quietly starts a replacement
    but the task it was running is lost, so ``imap`` waits forever. This
    actually happened on the full wdoor run: one shard's worker crashed
    and the job hung for hours.

    ``ProcessPoolExecutor`` instead marks itself "broken" and fails every
    unfinished future with ``BrokenProcessPool``. We collect those tasks
    and run them again on a fresh pool. Re-running is safe because shards
    are written atomically (a crashed worker leaves at most a ``.tmp``
    directory, which ``write_shard`` replaces). Ordinary Python
    exceptions raised by ``_build_shard`` are not retried: those are
    deterministic bugs and propagate to the caller as before.
    """
    pending = tasks
    for attempt in range(max_pool_restarts + 1):
        crashed = []
        with ProcessPoolExecutor(num_workers, initializer=_init_worker) as pool:
            # Look up _build_shard at call time (not import time) so tests
            # can substitute a worker that crashes.
            futures = {pool.submit(_build_shard, task): task for task in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except BrokenProcessPool:
                    crashed.append(futures[future])
                    continue
                record(result)
        # A crash fails every in-flight task, not just the one that caused
        # it. Some of those may have finished writing their shard just
        # before the pool broke (only the result message was lost). Count
        # them as done from their meta.json instead of rebuilding them:
        # renaming a rebuilt shard onto the finished one would fail.
        pending = []
        for task in sorted(crashed, key=lambda t: t[0]):
            finished = _finished_shard_summary(task)
            if finished is not None:
                record(finished)
            else:
                pending.append(task)
        if not pending:
            return
        shard_ids = [t[0] for t in pending]
        if attempt < max_pool_restarts:
            logger.warning(
                "A worker process died; retrying %d unfinished shard(s) on a "
                "fresh pool (restart %d/%d): %s",
                len(pending), attempt + 1, max_pool_restarts, shard_ids,
            )
    raise RuntimeError(
        f"Worker processes kept dying after {max_pool_restarts} pool "
        f"restart(s); shard(s) {shard_ids} were not built. Re-running the "
        "same command resumes from the finished shards. If the same shard "
        "always crashes, try --workers 1 to see the error."
    )


# ============================================================================
# Reading: PackedDataset
# ============================================================================


class PackedDataset(Dataset):
    """Map-style dataset over a packed directory, decoded batch-at-a-time.

    Arrays are opened with ``np.load(..., mmap_mode="r")``: the OS pages
    data in from disk on demand, so even a dataset much larger than RAM
    works, and nothing is read up front.

    **Batched loading.** PyTorch's DataLoader calls
    ``dataset.__getitems__(indices)`` with a whole batch of indices when
    the dataset defines it. We use that to decode the batch with a few
    vectorized NumPy calls instead of 1,024 separate ``__getitem__``
    calls. Because ``__getitems__`` already returns a finished batch,
    the DataLoader must be given :meth:`collate_fn` (an identity
    function) instead of its default collate; ``BonanzaTrainer`` picks
    it up automatically.

    Usage:

    .. code-block:: python

        ds = PackedDataset("data/packed/wdoor")
        loader = DataLoader(ds, batch_size=1024, shuffle=True,
                            num_workers=4, collate_fn=ds.collate_fn)
    """

    def __init__(self, root: str | Path, shard_ids: Optional[list[int]] = None):
        """Open a packed dataset.

        Args:
            root: Directory written by :func:`preprocess_csa_files`.
            shard_ids: Optional subset of shard numbers to use (the
                ``NNNNN`` in ``shard_NNNNN``). ``None`` uses every shard.
                Mostly used through :meth:`split`.
        """
        self.root = Path(root)
        config_path = self.root / CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found — is {self.root} a packed dataset?")
        self.config = json.loads(config_path.read_text())
        if self.config.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{self.root} has format_version={self.config.get('format_version')}, "
                f"this code reads version {FORMAT_VERSION}. Re-run preprocessing."
            )

        # Only completed shards are named exactly shard_NNNNN (see
        # write_shard); leftover shard_NNNNN.tmp directories from an
        # interrupted run are ignored even if their meta.json was written.
        self._shard_dirs: list[Path] = []
        sizes: list[int] = []
        for meta_path in sorted(self.root.glob(f"{SHARD_PREFIX}*/{SHARD_META_FILE}")):
            if not _SHARD_DIR_RE.fullmatch(meta_path.parent.name):
                continue
            if shard_ids is not None and _shard_id(meta_path.parent) not in shard_ids:
                continue
            meta = json.loads(meta_path.read_text())
            if meta["num_samples"] == 0:
                continue
            self._shard_dirs.append(meta_path.parent)
            sizes.append(meta["num_samples"])
        if not sizes:
            raise ValueError(f"No non-empty shards found in {self.root}.")

        # _starts[k] = global index of shard k's first sample; the extra
        # last entry is the total, so shard k covers [_starts[k], _starts[k+1]).
        self._starts = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self._starts[1:])

        # Opened lazily (per process) — see __getstate__.
        self._shards: Optional[list[dict[str, np.ndarray]]] = None

    @property
    def num_shards(self) -> int:
        return len(self._shard_dirs)

    @property
    def shard_ids(self) -> list[int]:
        """Shard numbers in use, in index order."""
        return [_shard_id(d) for d in self._shard_dirs]

    def split(self, val_fraction: float, seed: int = 0) -> tuple["PackedDataset", "PackedDataset"]:
        """Split into ``(train, validation)`` datasets by whole shards.

        Each shard holds complete games, so splitting by shard keeps all
        positions of a game on the same side. A position-level random
        split would put near-identical positions from one game into both
        sides and make validation look better than it really is.

        The validation shards are chosen at random (seeded, so the split
        is reproducible across runs and resumes). Because the input file
        list is sorted by year, random shards also spread validation
        across all years.

        Args:
            val_fraction: Fraction of shards to hold out, in (0, 1). At
                least one shard is always held out, and at least one is
                always kept for training.
            seed: RNG seed for choosing the validation shards.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if self.num_shards < 2:
            raise ValueError("Need at least 2 shards to split into train and validation.")
        ids = self.shard_ids
        n_val = min(max(1, round(val_fraction * len(ids))), len(ids) - 1)
        rng = np.random.default_rng(seed)
        val_ids = sorted(int(i) for i in rng.choice(ids, size=n_val, replace=False))
        train_ids = [i for i in ids if i not in set(val_ids)]
        return PackedDataset(self.root, train_ids), PackedDataset(self.root, val_ids)

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getstate__(self) -> dict:
        # DataLoader workers may receive a pickled copy of the dataset.
        # Pickling a memmap would copy the whole array into the message,
        # so drop the open arrays and let each worker reopen them.
        state = self.__dict__.copy()
        state["_shards"] = None
        return state

    def _open(self) -> list[dict[str, np.ndarray]]:
        if self._shards is None:
            self._shards = [
                {name: np.load(d / f"{name}.npy", mmap_mode="r") for name in _SHARD_ARRAYS}
                for d in self._shard_dirs
            ]
        return self._shards

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Decode a whole batch. Called by DataLoader with a list of indices."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError(f"Index out of range for dataset of size {len(self)}.")
        b = idx.size
        shards = self._open()

        planes = np.empty((b, PLANE_BYTES), dtype=np.uint8)
        scalars = np.empty((b, NUM_SCALAR_PLANES), dtype=np.float32)
        target = np.empty(b, dtype=np.uint16)
        value = np.empty(b, dtype=np.int8)
        legal_lists: list[np.ndarray] = [None] * b  # type: ignore[list-item]

        # Find which shard each index lives in, then gather shard by shard.
        shard_of = np.searchsorted(self._starts, idx, side="right") - 1
        for s in np.unique(shard_of):
            pos = np.nonzero(shard_of == s)[0]         # positions within the batch
            local = idx[pos] - self._starts[s]         # row numbers within shard s
            arr = shards[s]
            planes[pos] = arr["planes"][local]
            scalars[pos] = arr["scalars"][local]
            target[pos] = arr["target"][local]
            value[pos] = arr["value"][local]
            offsets = arr["legal_offsets"]
            legal = arr["legal_indices"]
            for p, row in zip(pos, local):
                legal_lists[p] = legal[offsets[row]:offsets[row + 1]]

        return unpack_batch(planes, scalars, target, value, legal_lists)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Single-sample access (unbatched tensors), for inspection and tests."""
        batch = self.__getitems__([index])
        return {k: v[0] for k, v in batch.items()}

    @staticmethod
    def collate_fn(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Identity collate: :meth:`__getitems__` already returns a batch."""
        return batch
