---
name: compact-disk-usage
description: Reclaim disk space after Bonanza preprocessing/training — survey data/ and checkpoints/, then (with confirmation) archive rebuildable artifacts with compression (packed shards, intermediate checkpoints) and remove extracted CSA files already covered by the .7z archives
---

Survey what the Bonanza pipeline left on disk, explain what each item is
and how to bring it back, then compact only what the user approves.

**Policy: archive with compression before deleting.** An item may be
removed outright only when a compressed copy of exactly the same content
already exists (e.g. extracted CSA files ↔ their `.7z` archives).
Everything else is first packed into a verified `.tar.zst` archive, and
only then is the original removed. Plain deletion without an archive is
a last resort that needs its own explicit confirmation.

Optional argument: `safe` (default — tiers 1–2) or `full` (also tier 3:
packed shards).

## Artifact map

Compression ratios measured on this repo's data with `zstd -9`:

| Tier | Path | Typical size | Action | Compressed | Restore cost |
|---|---|---|---|---|---|
| keep | `data/wdoor*.7z` | ~3.6 GB | **never touch** | — | — |
| keep | `data/wdoor_all_files.txt`, `data/wdoor_files.txt` | few MB | **never touch** (sorted lists make re-preprocessing reproduce identical shards) | — | — |
| 1 | `data/wdoor/` (extracted CSA) | ~25 GB | remove — already archived as `data/wdoor<year>.7z` | 3.6 GB (existing) | minutes (`7z x`) |
| 2 | intermediate checkpoints: `checkpoints/<run>/bonanza_step_*.pt`, all but the last `bonanza_epoch_*.pt`, older runs' `checkpoints/epoch_*.pt` | ~37–58 MB each | archive → `checkpoints/archive/<run>-intermediate.tar.zst` | **~87–92%** — float weights and Adam optimizer state are near-random bytes; saves little (2.63 → 2.41 GB for 72 v2 step checkpoints) | seconds |
| 3 | `data/packed/<name>/` (packed shards) | ~94 GB for full wdoor | archive → `data/archive/packed-<name>.tar.zst` | **~6–7% (~6 GB)** — consecutive positions in a game are nearly identical | ~10 min decompress vs ~18 h re-preprocessing |

Always keep uncompressed, per training run: `bonanza_best.pt` and the
highest-numbered `bonanza_epoch_*.pt`. Never touch `model.nnue`,
`*.onnx`, or anything tracked by git.

Tier 3 note: archive packed shards only when no training run will read
them soon — training needs them uncompressed (memory-mapped). If they
are still needed but `/home` is tight, moving them to another SSD is an
alternative; offer both.

## Steps

1. **Check nothing is still using the data.** Run
   `pgrep -af 'preprocess_kifu|train_bonanza|train_alphazero|fetch_and_pack'`.
   If anything matches, report it and stop: archiving or removing the
   inputs or checkpoints of a running job breaks it.
2. **Survey.** From the repo root run:
   ```bash
   df -h .
   du -shc data/*.7z 2>/dev/null | tail -1
   du -sh data/wdoor data/packed/* data/archive/* 2>/dev/null
   du -sh checkpoints/* 2>/dev/null
   ```
   and list checkpoint files per run directory (`ls -la checkpoints/<run>/`).
3. **Verify preconditions per tier:**
   - Tier 1: every year that has a `data/wdoor/<year>/` directory must
     also have `data/wdoor<year>.7z`, and `7z t data/wdoor<year>.7z`
     must pass. Years without a good archive are archived like tier 2/3
     instead (`tar -cf - -C data/wdoor <year> | zstd -9 -T0 -o data/archive/wdoor-<year>.tar.zst`).
   - Tier 2: the run must have a `bonanza_best.pt` or a final epoch
     checkpoint that stays uncompressed. Suggest exporting the best model
     first if no `.onnx` exists for it:
     `python scripts/alphazero/export_onnx.py --checkpoint checkpoints/<run>/bonanza_best.pt --output <run>.onnx --num-blocks N --num-filters F --value-head-hidden H`.
     The architecture flags default to the v1 net (4/64/64) and **must
     match the run's config** — for v2 that is `--num-blocks 10
     --num-filters 128 --value-head-hidden 128` (see the `model:`
     section of the config the run used).
   - Tier 3: confirm preprocessing finished (log ends without errors and
     shard count matches `ceil(lines in file list / files_per_shard)`
     from the packed dir's `config.json`), and check free space for the
     archive (~7% of the directory size).
4. **Present a plan** as a table: path, current size, action (remove /
   archive / move / keep), expected size afterwards, how to restore, and
   the total space freed. Ask the user to confirm (AskUserQuestion,
   multiSelect over the tiers/directories). Do nothing before
   confirmation.
5. **Archive, verify, then remove** — for each confirmed tier 2/3 item:
   1. Create the archive, listing files explicitly:
      ```bash
      mkdir -p data/archive
      tar -cf - -C data/packed <name> | zstd -9 -T0 -o data/archive/packed-<name>.tar.zst
      # checkpoints: tar -cf - -C checkpoints/<run> <file> <file> ... | zstd -9 -T0 -o checkpoints/archive/<run>-intermediate.tar.zst
      ```
      Use `-9` (fast, near-max ratio here); `-19` is ~40× slower for a
      few percent more.
   2. Verify: `zstd -t <archive>` passes, and the archive's file list
      matches the originals
      (`diff <(zstd -dc <archive> | tar -tf - | grep -v '/$' | sort) <(cd <parent> && find <name> -type f | sort)`
      for a directory; compare against the explicit file list for
      checkpoints). If verification fails, delete the bad archive, keep
      the originals, and report.
   3. Only after verification, remove the originals by explicit path
      (never `rm -rf data` or `rm -rf checkpoints`).
   For tier 1, remove `data/wdoor/<year>/` only for years whose `.7z`
   passed `7z t`.
6. **Report** the new `df -h .`, the archives created (with sizes), and
   the restore commands below.

## Restore commands (include in the final report)

```bash
# Extracted CSA files (tier 1)
for a in data/wdoor20*.7z; do y=$(basename "$a" .7z); y=${y#wdoor}; 7z x -o"data/wdoor/$y" "$a"; done

# Packed shards (tier 3) — ~10 minutes
zstd -dc data/archive/packed-<name>.tar.zst | tar -xf - -C data/packed

# Intermediate checkpoints (tier 2)
zstd -dc checkpoints/archive/<run>-intermediate.tar.zst | tar -xf - -C checkpoints/<run>

# Last resort if a packed archive is lost — re-preprocess (~18 h) from the
# same sorted file list for identical shards
python scripts/bonanza/preprocess_kifu.py --data-list data/wdoor_all_files.txt \
    --output data/packed/wdoor_all --min-ply 1 --workers 10
```

## Rules

- Archive before delete. Remove an original only after its compressed
  copy is verified, or when an equivalent verified archive already
  exists (tier 1 `.7z`).
- Deleting without an archive requires a separate, explicit
  confirmation naming the path; approval for one tier does not extend
  to another.
- Never touch `.7z` archives, file lists, archives under `*/archive/`,
  final/best checkpoints, or git-tracked files.
- Moving instead of compressing (e.g. packed shards to a second NVMe
  drive) is a valid option when the data will be trained on again soon.
- If a path in the map does not exist, skip it silently in the plan.
