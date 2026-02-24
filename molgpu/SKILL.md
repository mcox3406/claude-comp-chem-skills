---
name: molgpu
description: Reference guide for the MolGPU shared GPU cluster. Machine specs, SSH setup, shared filesystem, conda/mamba, multi-GPU training tips, and common commands.
---

# MolGPU Cluster Reference

Shared GPU cluster for the group. All nodes on private network (use MIT VPN to connect off campus).

## Machines

| Node | CPUs | GPUs | VRAM/GPU | RAM | Storage |
|------|------|------|----------|-----|---------|
| molgpu01 | 64 cores (2x AMD 3970X) | 2x RTX 3090 Ti | 24 GB | — | — |
| molgpu02 | 128 cores (2x AMD 3995WX) | 4x RTX A5000 | 24 GB | — | — |
| molgpu03 | 64 cores (AMD 3995WX) | 4x RTX A5000 | 24 GB | 512 GB | 2x 2TB SSD |
| molgpu04 | 64 cores (AMD 3995WX) | 3x RTX 3090 | 24 GB | 512 GB | 1TB + 2TB SSD (encrypted at /data) |
| molgpu05 | 32 cores (AMD 5975WX) | 2x RTX 4090 | 24 GB | 256 GB | 4TB SSD |
| molgpu06 | 32 cores (AMD 5975WX) | 2x RTX 4090 | 24 GB | 256 GB | 4TB SSD |
| molgpu07 | 128 cores (AMD 5995WX) | 4x RTX 4090 | 24 GB | 512 GB | 2TB SSD |
| molgpu08 | 52 cores (2x Xeon 6230R) | 8x RTX 2080 Ti | 11 GB | 512 GB | 1TB SSD |
| moldata01 | — (Synology DS3622xs) | — | — | — | 100TB HDD + 800GB SSD cache |

## Quick Selection Guide

- **4x GPU (fastest):** molgpu07 (4x 4090, best single-GPU perf + most cores)
- **4x GPU (fallback):** molgpu02 or molgpu03 (4x A5000)
- **2x GPU:** molgpu05 or molgpu06 (2x 4090)
- **Many small jobs:** molgpu08 (8x 2080 Ti, only 11 GB each)

## Access & SSH Setup

Full hostnames are `molgpu01.mit.edu` through `molgpu08.mit.edu`. Same credentials across all nodes. SSH keys set up on one node work on all others (shared home dir).

Add this to `~/.ssh/config` for easy access:

```
Host *
  AddKeysToAgent yes
  UseKeychain yes   # macOS only — saves key password in keychain
  ForwardAgent yes

Host molgpu0?
  Hostname %h.mit.edu
  User <your-username>
  ForwardAgent yes
  IdentityFile <path/to/private_key>
```

This lets you `ssh molgpu07` instead of `ssh user@molgpu07.mit.edu -i /path/to/key`. `ForwardAgent` means you don't need to store your private key on the server to use GitHub via SSH.

To add access from a new client, append your public key to `~/.ssh/authorized_keys` on any molgpu node.

Change password with `passwd` (applies to all nodes, must be mixed case + numeric, 8+ chars).

## Shared Filesystem

All molgpu machines mount the same NFS home directory at `/mnt/home/` (81.8 TB total). Files written on one machine are immediately visible on all others — **no rsync needed between molgpus**. Only rsync from your local machine to any one node.

Sensitive data goes on the encrypted partition at `/mnt/encrypted/` (16.4 TB). Protect sensitive folders with `chmod 700`.

**Important:** Avoid reading/writing a large number of files in parallel — this slows NFS for everyone. Reduce parallel I/O workers if NFS feels slow.

**Always back up code and important checkpoints.** There is no guarantee storage will never fail.

## Conda / Mamba

Global conda is available on molgpu02–06 (no need to install your own):
```bash
/opt/miniconda3/bin/conda init
```

Global mamba (faster drop-in replacement for conda) on molgpu02:
```bash
# after conda init:
mamba init
```

`uv` is also available at `~/.local/bin/uv` if installed per-user.

## Dashboard

View current load across all machines: http://molgpu01:8088/

## Common Commands

```bash
# transfer files from local machine (only need to hit one node, shared fs does the rest)
rsync -az --exclude '.venv*' --exclude '__pycache__' --exclude '.git' \
  . molgpu07:~/project-name/

# check GPU availability on a node
ssh molgpu07 nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# scan all machines at once
for h in molgpu0{1..8}; do echo "=== $h ===" && ssh $h nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null; done

# always use tmux for long-running jobs (nohup is NOT enough — child processes get SIGHUP)
ssh molgpu07
tmux new-session -s myjob
# ... launch job ...
# detach: Ctrl-B then D
# reattach later: tmux attach -t myjob

# always redirect stderr (some frameworks log errors only to stderr)
python train.py > /tmp/train.log 2>&1

# kill orphaned GPU processes after a crash
nvidia-smi  # find PIDs
kill -9 <pid>
```

## Known Issues

- **NFS stale file handles.** Caches stored in home dir (`~/.triton/cache`, `~/.cache/torch/`) can go stale when switching between machines. Fix: use local disk for caches (`export TRITON_CACHE_DIR=/tmp/triton_cache`) or nuke the cache (`rm -rf ~/.triton/cache`).
- **All GPUs are 24 GB** (except molgpu08 at 11 GB).

## LLM/VLM-Specific Notes

### Multi-GPU Training (DDP)

```bash
# set HF_HUB_OFFLINE=1 to avoid HF Hub race conditions during multi-GPU load
# (download model first, then set this)
HF_HUB_OFFLINE=1 accelerate launch --num_processes=4 train.py

# QLoRA + DDP: use device_map={"": local_rank} (not None or "auto")
```

### vLLM Inference on 24 GB GPUs

VLMs are especially memory-hungry. If OOM during warmup or generation, tune:
- `gpu_memory_utilization`: lower from 0.9 to 0.35–0.7
- `max_num_seqs`: lower from 256 to 4–16
- `enforce_eager=True`: skips CUDA graphs, saves ~2 GB VRAM
- `tensor_parallel_size`: split model across GPUs

### Model Size Reference

| Size | bf16 VRAM | QLoRA NF4 | Min GPUs (inference) |
|------|-----------|-----------|---------------------|
| 4B | ~8 GB | ~5 GB | 1 |
| 8B | ~16 GB | ~8-10 GB | 1 |
| 12B | ~24 GB | ~12-14 GB | 2 |
| 70B | ~140 GB | ~42 GB | 8+ |
