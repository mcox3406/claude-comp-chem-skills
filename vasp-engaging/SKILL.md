---
name: vasp-engaging
description: VASP on MIT Engaging (using Rafa's group's version). CentOS 7 constraint, module loading, SLURM submission, POTCAR paths, and known pitfalls. Use when setting up or running VASP calculations on Engaging.
---

# VASP on MIT Engaging

VASP 6.2.1 is compiled against Intel MKL/MPI 2017 (CentOS 7). It **cannot run on Rocky 8 nodes**. All VASP jobs must target CentOS 7 compute nodes using `-C centos7` in the SLURM script.

## Major Constraint: CentOS 7 Only

The cluster is migrating to Rocky 8. Most group partitions (ou_cheme, pi_ccoley) now have **only Rocky 8 nodes**. VASP will fail with `execve(): vasp_std: No such file or directory` if it lands on a Rocky 8 node.

**Always add `#SBATCH -C centos7`** to VASP job scripts.

## Partitions with CentOS 7 Nodes

| Partition | CentOS 7 Nodes | Access |
|-----------|---------------|--------|
| `sched_mit_rafagb` | node1034-1035, node1115-1130, node1243-1258, node1347-1362 (~40 nodes) | Gomez-Bombarelli group (also gives POTCAR access) |
| `sched_mit_ccoley` | node1236, node1238 | Coley group |
| `sched_mit_hill` | node073-node389 (~60 nodes) | Open |
| `sched_any` | node087, node122, node161, node170, node235 | Open |
| `newnodes` | node119-node425 (~25 nodes) | Open |

**Recommended:** Use `sched_mit_rafagb`: reliable CentOS 7 nodes, same group that owns the POTCARs.

## Module Loading Sequence

This exact sequence is required in every SLURM script:

```bash
source ~/.bashrc
export VASP=vasp_std    # or vasp_gam for gamma-only
module purge
module use -a /nfs/rafagblab001/software/modulefiles
module load vasp/6.2.1
```

The `module load vasp/6.2.1` depends on `intel/2017-01`, which is **only available on CentOS 7 nodes**. It will fail on Rocky 8 with `The following module(s) are unknown: "intel/2017-01"`.

Binary location: `/nfs/rafagblab001/software/vasp/vasp.6.2.1/bin/vasp_std`

## VASP Variants

| Binary | Use Case |
|--------|----------|
| `vasp_std` | Standard: general k-point meshes |
| `vasp_gam` | Gamma-only: molecules/clusters in box (~2x faster at gamma) |
| `vasp_ncl` | Non-collinear: SOC calculations |

## POTCAR Location

POTCARs (PBE, v54) are at `/nfs/rafagblab001/vasp_PBE_54/`. Access requires membership in the `sched_mit_rafagb` group (owned by `dskoda`).

To build a POTCAR:
```bash
cat /nfs/rafagblab001/vasp_PBE_54/Ag/POTCAR \
    /nfs/rafagblab001/vasp_PBE_54/O/POTCAR \
    > POTCAR
```

For `_pv` or `_sv` variants (more valence electrons):
```bash
cat /nfs/rafagblab001/vasp_PBE_54/Na_pv/POTCAR > POTCAR
```

Species order in POTCAR **must match** species order in POSCAR.

## SLURM Submission Template

```bash
#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -C centos7
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 0-06:00:00
#SBATCH -J vasp_job
#SBATCH --output=%j_%x.log

source ~/.bashrc
module purge
module use -a /nfs/rafagblab001/software/modulefiles
module load vasp/6.2.1

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
srun --mpi=pmi2 /nfs/rafagblab001/software/vasp/vasp.6.2.1/bin/vasp_std
```

Key points:
- **`-C centos7` is mandatory** -- without it, jobs will fail on Rocky 8 nodes
- **`-p sched_mit_rafagb`** is the best partition for VASP (CentOS 7, same group as POTCARs)
- **`srun --mpi=pmi2`** is required -- plain `srun` conflicts with Intel MPI's wrapper, and `mpirun` is not available on compute nodes. The `--mpi=pmi2` flag tells SLURM to use PMI2 for process launching.
- **Use full path to binary** (`/nfs/rafagblab001/software/vasp/vasp.6.2.1/bin/vasp_std`) -- the module adds it to PATH but using the full path is more reliable
- Always `--nodes=1` to avoid cross-node MPI issues
- `OMP_NUM_THREADS=1` prevents VASP from spawning excessive threads
- Run in `$SLURM_SUBMIT_DIR` directly -- no need to copy to scratch

## Typical Resource Requirements

(Will add in later commit as I run more calculations)

## Parallelization Settings (INCAR)

(Will add in later commit as I run more calculations)

## File Transfer

`scp` works but `rsync` may hang due to Duo 2FA. For directories, create the remote dir first:

```bash
ssh engaging1 "mkdir -p ~/path/to/remote/dir"
scp local/files/* engaging1:~/path/to/remote/dir/
```

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `execve(): vasp_std: No such file or directory` | Job ran on Rocky 8 node | Add `#SBATCH -C centos7` |
| `module(s) are unknown: "intel/2017-01"` | Module loaded on Rocky 8 | Only load on CentOS 7 nodes/login |
| `mpirun: command not found` | Intel MPI not in PATH on compute node | Use `srun --mpi=pmi2` instead |
| MPI help text dump in log, VASP doesn't run | `srun` picks up Intel MPI's `mpiexec.slurm` wrapper | Use `srun --mpi=pmi2` with full path to binary |
| `Permission denied` on POTCARs | Not in `sched_mit_rafagb` group | Ask admin to add you |
| `rsync` hangs | Duo 2FA issue | Use `scp` instead (create remote dirs first) |
| POTCAR mismatch | Species order wrong | Match POSCAR species order exactly |
| VASP 6.4+ INCAR tags fail | Binary is 6.2.1 | Don't use ML_LMLFF, etc. |
