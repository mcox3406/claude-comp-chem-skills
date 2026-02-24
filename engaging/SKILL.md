---
name: engaging
description: MIT Engaging cluster reference — SLURM job scheduling, Coley group nodes, filesystem layout, Python environments, and common commands. Use when working with the Engaging HPC cluster.
---

# MIT Engaging Cluster Reference

SLURM-managed HPC cluster run by MIT ORCD. Duo 2FA required at each login.

*Tips adapted from the Coley Group Engaging Guide by Sam Goldman, David Graff, Runzhong Wang, and Magdalena Lederbauer.*

## Access

Login nodes (Rocky 8):
```
ssh username@orcd-login001.mit.edu   # through orcd-login004
```

SSH config for convenience:
```
Host engaging
  Hostname orcd-login001.mit.edu
  User <kerberos>
  ForwardAgent yes
```

## Coley Group Nodes

| Partition | Node | CPUs | GPUs | GPU VRAM | RAM |
|-----------|------|------|------|----------|-----|
| sched_mit_ccoley | node1236 | 52 (2x Intel 6230R) | 8x 2080 Ti | 11 GB | 512 GB |
| sched_mit_ccoley | node1237 | 52 (2x Intel 6230R) | 8x 2080 Ti | 11 GB | 512 GB |
| sched_mit_ccoley | node1238 | 128 (2x AMD EPYC 7702) | — | — | — |
| pi_ccoley | node2436 | 128 (2x Intel 8562Y+) | 4x H100 | 80 GB | — |
| pi_ccoley | node2519-2522 | 128 (2x AMD EPYC 7513) | — | — | — |

**Resource etiquette for node1236/1237:** 52 CPUs and 512 GB across 8 GPUs. Use `-n 6 --mem-per-cpu=8G` per single-GPU job so all 8 GPUs can run simultaneously. Requesting too many CPUs or too much memory per job blocks other GPUs from being used.

Prefer CPU-only nodes (node1238, node2519-2522) for non-GPU work.

## Public Partitions

| Partition | Purpose | GPUs | Max Time |
|-----------|---------|------|----------|
| mit_normal | CPU batch/interactive | — | 12 hrs |
| mit_normal_gpu | GPU jobs | L40S (44 GB), H100 (80 GB), H200 (140 GB) | 6 hrs |
| mit_quicktest | Quick testing | — | 15 min |
| mit_preemptable | Low-priority, may be preempted | A100/L40S/H100/H200 | 48 hrs |

Request GPUs with `-G [type]:[count]` (e.g., `-G l40s:1`, `-G h200:2`). Default on mit_normal_gpu is L40S.

CPUs per GPU on mit_normal_gpu: L40S = 16, H200 = 15. Requesting more CPUs than this wastes a GPU slot.

## Filesystem

| Path | Size | Backed Up | Purpose |
|------|------|-----------|---------|
| `/home/<user>` | 200 GB | Yes (snapshots) | Code, config, small files |
| `/home/<user>/orcd/pool` | 1 TB | No | Larger datasets (not I/O intensive) |
| `/home/<user>/orcd/scratch` | 1 TB | No | I/O-heavy job data (flash storage) |
| `/nfs/ccoleylab001` | 10 TB shared | No | Group shared data — don't be greedy |
| `/nobackup` | cluster-global | No | Semi-persistent intermediate files (periodically wiped) |
| `/tmp` | node-local | No | Fastest I/O, but not shared between nodes; gone after job ends |

**Storage hierarchy for jobs:** Use `/tmp` or scratch for active computation. Stage data from pool. Keep code and configs in home.

Scratch files deleted after 6 months of inactivity. `/nobackup` wiped ~monthly.

## SLURM Quick Reference

```bash
# check partition availability
sinfo -p sched_mit_ccoley
sinfo -p mit_normal_gpu -O Partition,Nodes,CPUs,Memory,Gres

# interactive session (1 GPU on group node)
salloc -p sched_mit_ccoley -w node1236 --gres=gpu:1 -n 6 --mem-per-cpu=8G --time=3:00:00

# interactive session (public H100)
salloc -p mit_normal_gpu -G h100:1 -c 4 --mem=32G --time=6:00:00

# batch job
sbatch myjob.sh

# check your jobs
squeue --me

# cancel a job
scancel <jobid>

# job history and memory usage
sacct -j <jobid> -o JobID,JobName,State,ReqMem,MaxRSS,Elapsed --units=G

# check node resources
scontrol show node node1236

# who's using our partition
squeue -p sched_mit_ccoley
```

### Example Batch Script

```bash
#!/bin/bash
#SBATCH -p sched_mit_ccoley
#SBATCH -w node1236
#SBATCH --gres=gpu:1
#SBATCH -n 6
#SBATCH --mem-per-cpu=8G
#SBATCH -t 0-08:00:00
#SBATCH -J my_job
#SBATCH --output=logs/%j_%x.log

module load miniforge
source activate myenv

python train.py
```

### Batch with Variable Command

```bash
#!/bin/bash
#SBATCH -p sched_mit_ccoley
#SBATCH -n 6
#SBATCH --mem-per-cpu=8G
#SBATCH -t 0-05:00:00
#SBATCH --output=logs/%j_%x.log

module load miniforge
source activate myenv
eval $CMD
```
Launch with: `sbatch --export=CMD="python train.py --lr 1e-4" generic_slurm.sh`

### Preemptable Jobs

Use `--requeue` flag so jobs restart when preempted. Ensure your code checkpoints regularly.
```bash
sbatch --requeue -p mit_preemptable -G l40s:1 myjob.sh
```

## Python Environment Setup

```bash
# load system miniforge (do NOT install your own conda/miniconda)
module load miniforge

# option 1: conda/mamba environment
mamba create -n myenv python=3.12 pytorch torchvision
source activate myenv
# install pip-only packages after activating:
pip install some_package   # do NOT use --user flag

# option 2: virtual environment
python -m venv ~/myproject/venv
source ~/myproject/venv/bin/activate
pip install -r requirements.txt   # do NOT use --user flag
```

**In job scripts**, always load miniforge and activate the environment explicitly — don't rely on login shell state:
```bash
module load miniforge
source activate myenv
```

### Things to Avoid

- **Do NOT** run `conda init` or `mamba init` — it modifies `.bashrc` and causes issues
- **Do NOT** install your own miniconda/miniforge/anaconda
- **Do NOT** use `pip install --user` — installs to `~/.local` instead of the environment
- **Do NOT** set `PYTHONPATH` unless absolutely necessary

### Conda Environments Filling Home?

Move them to scratch by creating `~/.condarc`:
```yaml
envs_dirs:
  - /home/USERNAME/orcd/scratch/.conda/envs
pkgs_dirs:
  - /home/USERNAME/orcd/scratch/.conda/pkgs
auto_activate_base: false
```
Back up environments with `conda export --name myenv --file=environment.yaml`.

## Jupyter with GPU

Port-forwarding approach (better GPU than OOD's default K20):
```bash
# terminal 1: get a GPU allocation
ssh orcd-login001.mit.edu
salloc -p sched_mit_ccoley -w node1236 --gres=gpu:1 -n 6 --mem-per-cpu=8G --time=3:00:00
# note the allocated node, e.g., node1236

# terminal 2: port forward through login node to compute node
ssh orcd-login001.mit.edu -L 9998:localhost:9998
ssh node1236 -L 9998:localhost:9998
jupyter lab --port 9998
```

## Copying Files

```bash
# from local machine to engaging
scp -r localdir/ username@orcd-login001.mit.edu:~/project/

# or with rsync (better for repeated syncs)
rsync -az --exclude '.venv*' --exclude '__pycache__' --exclude '.git' \
  . username@orcd-login001.mit.edu:~/project/
```

## Troubleshooting

- **Python can't find packages in job:** Check `which python` points to your env. Load miniforge + activate env in the job script, not just at login.
- **Conda activate fails in batch script:** Add `eval "$(conda shell.bash hook)"` before `conda activate`.
- **Home directory full:** Check conda envs (`du -sh ~/.conda`), move to scratch (see above). Remove unused envs with `conda remove -n old_env --all`.
- **Job pending (Resources):** Requested resources not available yet. Check `sinfo` for node states.
- **Job pending (Priority):** Someone else is ahead in queue. Consider mit_preemptable for longer jobs.
- **Wrong python version:** `which python` to verify. Virtual envs inherit the python used to create them.
