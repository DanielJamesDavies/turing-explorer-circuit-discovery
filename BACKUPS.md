# Backups

Where every irreplaceable artefact lives (DAN-6). Update this file whenever a backup is made or a location changes.

## Locations

| What | Primary | Second copy |
|---|---|---|
| Code, tracked experiment scripts and small results, `paper/` | this repo (`X:`) | GitHub `origin/multi-device` (NOT `main`, which is stale); last push `0b18c6e`, 2026-09-28 |
| Everything below, snapshot of 2026-10-07 | `X:\Projects\AIs\Turing\Publication\3 Implementation\` | `G:\Paper Backup Files\2026-10-07\` (robocopy logs beside each folder) |
| 15k protocol-v1 run (`experiments/062-h100-protocol-v1/out_full`) | `X:` | `G:` snapshot; also a tgz on the RunPod network volume, if that volume still exists |

## The 2026-10-07 snapshot on G: (121 GB, 0 failed files)

| Folder | Source | Files | Size | Contents |
|---|---|---|---|---|
| `experiments` | `2\experiments` | 52,351 | 51.5 GB | every experiment, including the 15k run, the sweep, the weak-penalty runs, the random null, the alpha = 1 re-scoring and the member-restoration data |
| `outputs` | `2\outputs` | 1,232 | 20.4 GB | discovery artefacts the pipeline loads (`candidates.pt` and others) |
| `Runs` | `..\Runs` | 6,890 | 17.1 GB | the context stores (top, mid-band, retrieval) |
| `models` | `2\models` | 37 | 14.1 GB | TuringLLM weights and the 36 SAEs (also in the sibling `sae-system` project, on the SAME drive) |
| `data` | `2\data` | 6,061 | 12.6 GB | the corpus shards; sequence ids are positional over the sorted shard list, so keep the set complete |
| `dev-notes` | `2\dev-notes` | 32,365 | 4.9 GB | notes and old run data; the Linux symlink `data/venv-ct/lib64` inside an old virtualenv could not be copied (replaceable) |
| `paper` | `2\paper` | 52 | 3.7 MB | the draft (also in git) |

`__pycache__` folders were excluded. The copy was made with `robocopy /E` (copy only, nothing deleted on either side).

## Not backed up

- `.venv/`: rebuild with `pip install -r requirements-cu128.txt` in WSL (see the environment recipe in the project memory / README).
- Anything written after 2026-10-07 until the next snapshot. Re-run the snapshot after large new runs:

```
robocopy "<source>" "G:\Paper Backup Files\<date>\<name>" /E /COPY:DAT /R:1 /W:1 /MT:8 /XD __pycache__ .venv /LOG:"G:\Paper Backup Files\<date>\robocopy-<name>.log"
```

robocopy exit codes 0-7 mean success (1 = files copied); 8 or more means some files failed.
