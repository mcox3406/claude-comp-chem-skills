---
name: vasp-engaging
description: VASP on MIT Engaging. VASP 6.4.2 (GCC/OpenMPI, Rocky 8 compatible), module loading, SLURM submission, POTCAR paths. Use when setting up or running VASP calculations on Engaging.
---

# VASP on MIT Engaging

## VASP 6.4.2 (Recommended)

Compiled with GCC 12.2.0 + OpenMPI 4.1.4 on AMD EPYC (Rocky 8). Works on all current partitions without CentOS 7 constraints.

Binary: `/nfs/rafagblab001/vasp.6.x.x/vasp.6.4.2/bin/vasp_std`
Module: `/nfs/rafagblab001/software/modulefiles/vasp/6.4.2`

### Module Loading

```bash
module use -a /nfs/rafagblab001/software/modulefiles
module load vasp/6.4.2
```

This automatically loads GCC 12.2.0, OpenMPI 4.1.4, and sets `LD_LIBRARY_PATH` for OpenBLAS, FFTW, and ScaLAPACK.

### SLURM Submission Template

```bash
#!/bin/bash
#SBATCH -p mit_normal           # or ou_cheme, mit_preemptable, pi_ccoley, etc.
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 0-06:00:00
#SBATCH -J vasp_job
#SBATCH --output=%j_%x.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use -a /nfs/rafagblab001/software/modulefiles
module load vasp/6.4.2

cd $SLURM_SUBMIT_DIR
srun vasp_std
```

### VASP Variants

| Binary | Use Case |
|--------|----------|
| `vasp_std` | Standard: general k-point meshes |
| `vasp_gam` | Gamma-only: molecules/clusters in box (~2x faster at gamma) |
| `vasp_ncl` | Non-collinear: SOC calculations |

## POTCAR Location

POTCARs (PBE, v54) are at `/nfs/rafagblab001/vasp_PBE_54/`. Access requires membership in the `sched_mit_rafagb` group.

```bash
cat /nfs/rafagblab001/vasp_PBE_54/Ag/POTCAR \
    /nfs/rafagblab001/vasp_PBE_54/O/POTCAR \
    > POTCAR
```

Species order in POTCAR **must match** species order in POSCAR.

## Build from Source

If you need to rebuild (e.g., for a new version), follow https://orcd-docs.mit.edu/recipes/build-vasp-gcc-cpu/:

```bash
# get an interactive session on an AMD EPYC node (where jobs will run)
salloc -p pi_ccoley -w node2519 -n 8 --mem=16G --time=1:00:00

module load gcc/12.2.0 openmpi/4.1.4 netlib-lapack/3.10.1 netlib-scalapack/2.2.0 fftw/3.3.10 openblas/0.3.26

cd /nfs/rafagblab001/vasp.6.x.x/vasp.6.4.2
cp arch/makefile.include.gnu_omp makefile.include

SCALAPACK_ROOT=$(module -t show netlib-scalapack 2>&1 | grep CMAKE_PREFIX_PATH | awk -F, '{print $2}' | awk -F\" '{print $2}')
FFTW_ROOT=$(pkgconf --variable=prefix fftw3)
OPENBLAS_ROOT=$(dirname $(pkgconf --variable=libdir openblas))

make -j 8 OPENBLAS_ROOT=$OPENBLAS_ROOT FFTW_ROOT=$FFTW_ROOT SCALAPACK_ROOT=$SCALAPACK_ROOT MODS=1 DEPS=1
```

**Important:** Always compile on the same CPU architecture where jobs will run. `-march=native` in the makefile bakes in the compile node's instruction set.

## File Transfer

`scp` works but `rsync` may hang due to Duo 2FA. For directories, create the remote dir first:

```bash
ssh engaging1 "mkdir -p ~/path/to/remote/dir"
scp local/files/* engaging1:~/path/to/remote/dir/
```

## VASP 6.2.1 (Legacy)

The old VASP 6.2.1 at `/nfs/rafagblab001/software/vasp/vasp.6.2.1/` is compiled against Intel MKL/MPI 2017 and **only works on CentOS 7 nodes** (`sched_mit_rafagb` with `-C centos7`). Requires `srun --mpi=pmi2` with the full binary path. Use 6.4.2 instead unless you specifically need 6.2.1.

The old submission script looks like:
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


## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `Illegal instruction` | Binary compiled on different CPU arch | Recompile on the target node type (see Build section) |
| `cannot open shared object file: libscalapack.so` | `LD_LIBRARY_PATH` not set | Use the module (`module load vasp/6.4.2`) which sets this |
| `Permission denied` on POTCARs | Not in `sched_mit_rafagb` group | Ask admin to add you |
| `rsync` hangs | Duo 2FA issue | Use `scp` instead |
| POTCAR mismatch | Species order wrong | Match POSCAR species order exactly |
