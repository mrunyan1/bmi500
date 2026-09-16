# %%

import time
start = time.time()
import socket
if socket.gethostname().startswith('oddjobs'):
    raise SystemExit('Use sbatch or srun to run this on a compute node.')

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import sys
import argparse
import cProfile
import csv
import os
import pstats

early_timings = [('imports', time.time() - start)]
start = time.time()

# %%
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
early_timings.append(('scanpy_settings', time.time() - start))
start = time.time()

# %%
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='A new output directory for this run', required=True)
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)
parser.add_argument('--profile-mode', choices=['coarse', 'instrumented', 'fine'], default='coarse')

args = parser.parse_args()
if args.num_threads < 1:
    parser.error('--num-threads must be at least 1')

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set 
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads

sc.settings.n_jobs = nthreads
print(f"using {sc.settings.n_jobs} Scanpy jobs")
os.makedirs(outdir)  # keep earlier runs
sc.settings.cachedir = outdir + 'cache'
early_timings.append(('arguments_and_paths', time.time() - start))

timing_file = outdir + 'section_times.csv'
if args.profile_mode == 'instrumented':
    with open(timing_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'section', 'seconds'])
        for section, seconds in early_timings:
            writer.writerow([dataset, section, seconds])
            print(f'{section}: {seconds:.6f} seconds', flush=True)


def log_time(section, t0):
    seconds = time.time() - t0
    if args.profile_mode == 'instrumented':
        print(f'{section}: {seconds:.6f} seconds', flush=True)
        with open(timing_file, 'a', newline='') as f:
            csv.writer(f).writerow([dataset, section, seconds])
    return time.time()


with open(outdir + 'run_info.txt', 'w') as f:
    f.write(f'dataset: {dataset}\nmode: {args.profile_mode}\n')
    f.write(f'host: {socket.gethostname()}\npython: {sys.version}\n')
    f.write(f'Scanpy jobs: {nthreads}\n')

start = time.time()

#%%

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    #'/nethome/tpan7/scgc/data/' + dataset + '/filtered_gene_bc_matrices/hg19',  # the directory with the `.mtx` file
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`


start = log_time('read_data', start)
print(f'Input: {adata.n_obs} cells, {adata.n_vars} genes', flush=True)
with open(outdir + 'run_info.txt', 'a') as f:
    f.write(f'input_cells: {adata.n_obs}\ninput_genes: {adata.n_vars}\n')
start = time.time()

# %%
# preprocessing

# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
start = log_time('filter', start)
print(f'After filtering: {adata.n_obs} cells, {adata.n_vars} genes', flush=True)
with open(outdir + 'run_info.txt', 'a') as f:
    f.write(f'filtered_cells: {adata.n_obs}\nfiltered_genes: {adata.n_vars}\n')
start = time.time()

#%%
# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
start = log_time('normalize_and_log', start)


# %%
# highly variable genes

#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# freeze data.
adata.raw = adata

# filtering by highly variable genes.
adata = adata[:, adata.var.highly_variable]
start = log_time('highly_variable_genes', start)


#%%
# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)
start = log_time('scale', start)

# %%
# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata
start = log_time('commented_report', start)

# %%
# pca.  parallel via OMP_NUM_THREADS
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
start = log_time('pca', start)

# adata.write(results_file)
# adata

# %%
# neighborhood graph
sc.pp.neighbors(adata, n_pcs=30)
start = log_time('neighbors', start)

# %% 
# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata
start = log_time('commented_paga', start)


# %%
# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
sc.tl.louvain(adata, resolution = 0.5)
start = log_time('louvain', start)


#%%
# umap
sc.tl.umap(adata, n_components=30)
start = log_time('umap', start)

#%%
adata.write(results_file)
adata
start = log_time('write_h5ad', start)

# %%
# support t-test, wilcoxon, logistic regression
# find marker genes
if args.profile_mode == 'fine':
    profile_file = outdir + 'rank_genes_groups.pstats'
    cProfile.run(
        "sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)",
        profile_file)
    pstats.Stats(profile_file).strip_dirs().sort_stats('cumulative').print_stats(30)
else:
    sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)
start = log_time('rank_genes_groups', start)

with open(outdir + 'completed.txt', 'w') as f:
    f.write(f'{dataset} {args.profile_mode}: completed\n')
print('RUN COMPLETED', flush=True)
