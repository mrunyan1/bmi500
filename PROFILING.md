# PBMC profiling

Edit and commit on the Mac. Matthew pushes to his fork. Pull that commit on the cluster.

```text
Mac files  ->  GitHub fork  ->  cluster checkout  ->  Slurm jobs
Mac report <-  logs, section timings, and cProfile results
```

## Cluster setup

Use the existing compute session, such as bignode5, for environment installation and extraction.
The checkout and its environment live in shared scratch, which is temporary storage.

```bash
cd /opt/scratchspace/mrunyan/bmi500_hw3
git clone https://github.com/mrunyan1/bmi500.git
cd bmi500
module load bmi/python-3.12.12
python -m venv .venv
source .venv/bin/activate
git pull --ff-only
python -m pip install -r requirements.txt
python -m pip check
python -c 'import scanpy, anndata, igraph, louvain; print("Imports OK")'
mkdir -p logs results
tar -xzf data/pbmc3k.tgz -C data
tar -xzf data/pbmc6k.tgz -C data
tar -xzf data/pbmc10k.tgz -C data
```

Skip the clone command if this checkout already exists. Keep the four original package
pins. AnnData 0.12.19 caused an import failure with the required typing_extensions 4.12.2
in the original Mac environment. The added AnnData pin avoids that dependency chain;
NumPy and pandas stay on their earlier major APIs. An import check on the cluster is
still required. Use a fresh environment rather than copying the Mac environment.

## Run

Submit from the checkout root so Slurm can find the logs directory. After environment
setup, exit the interactive compute shell to return to OddJobs before submitting.

```bash
cd /opt/scratchspace/mrunyan/bmi500_hw3/bmi500
mkdir -p logs results
sbatch slurm/pbmc3k_coarse.sbatch
squeue -u mrunyan
```

Check that this first job succeeds before submitting the other eight:

```bash
sbatch slurm/pbmc3k_instrumented.sbatch
sbatch slurm/pbmc3k_fine.sbatch
sbatch slurm/pbmc6k_coarse.sbatch
sbatch slurm/pbmc6k_instrumented.sbatch
sbatch slurm/pbmc6k_fine.sbatch
sbatch slurm/pbmc10k_coarse.sbatch
sbatch slurm/pbmc10k_instrumented.sbatch
sbatch slurm/pbmc10k_fine.sbatch
```

Every script requests one CPU, 8 GB, and one hour on overflow. It uses a login bash
shell to load the Python module. It records the hostname, Git commit, Python version,
and installed packages. Library thread limits are set before Python starts.

## What changed

The analysis calls and parameters in scanpy_pbmc.py are retained, including the
30 component UMAP and writing the h5ad file before marker testing.

* Coarse: /usr/bin/time -v measures the whole process. Its output is in the Slurm log.
* Instrumented: time.time() measures each of the 16 original sections. Timings are
  printed and appended to section_times.csv. The two sections containing only
  comments remain visible and should not be interpreted as analysis work.
* Fine: cProfile.run() measures only sc.tl.rank_genes_groups. The preceding pipeline
  runs normally to prepare adata. The profile is saved and its top 30 cumulative
  entries are printed. Profiling adds overhead, so use the instrumented measurements
  for the runtime growth plot.

The unused thread argument now sets sc.settings.n_jobs after argument parsing.
This alone does not constrain every native library, so the batch scripts also set
the numerical library thread limits. All nine jobs use one CPU.

Each job gets a new output directory, Scanpy cache, Numba cache, and Matplotlib cache.
This includes application cache creation and first compilation in the measurements.
The operating system disk cache is not cleared. Record the node names when comparing
runs because the cluster may place them on different hardware.

## Results

Logs are in logs/DATASET_MODE_JOBID.out. The standard error stream is included there.
Results are in results/DATASET_MODE_JOBID/.

* run_info.txt records the mode, hostname, and actual cell and gene counts.
* section_times.csv exists for instrumented runs.
* rank_genes_groups.pstats exists for fine runs.
* completed.txt is written only after the pipeline finishes successfully.
* The h5ad files and caches remain on scratch; they are not needed for the report.

Interrupted jobs can leave partial logs and timings. Only use runs with completed.txt
and a successful Slurm exit state. Rerun a failed or preempted job with sbatch to get
a new job ID and output directory. Do not combine partial runs as a completed result.

Retrieve the small evidence files to the Mac before scratch cleanup. Preserve the
logs, CSVs, pstats files, run_info.txt, and completed.txt; leave h5ad and cache files
on scratch. Git ignores logs, results, extracted data, and the virtual environment.
The original three course archives are already tracked in the repository.

Runtime figures and the 20K extrapolation require successful cluster measurements.
No measured runtimes or extrapolated values have been supplied by these edits.
