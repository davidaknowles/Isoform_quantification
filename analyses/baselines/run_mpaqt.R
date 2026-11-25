# MPAQT 2.4.0 on SG-NEx: short reads (kallisto bus equivalence classes, prepared by MPAQT) jointly
# with long-read transcript counts. MPAQT does not count long reads itself except via genome
# alignment + Bambu, so the long-read counts are lr-kallisto's estimated counts (run_lr_kallisto.sh).
# Defaults otherwise: no positional bias, default prior, TPM output.
#   index:            build the MPAQT index (kallisto index + P matrix for 150 bp mates) once
#   rep <cell> <rep>: prepare the replicate's Illumina reads, then quantify with each ONT protocol
#
# Run (through run_mpaqt.sh):
#   sbatch analyses/baselines/run_mpaqt.sh index
#   sbatch --array=1-6 analyses/baselines/run_mpaqt.sh rep
.libPaths(c("/gpfs/commons/groups/knowles_lab/daknowles/JOLI/tools/Rlib", .libPaths()))
suppressPackageStartupMessages({library(mpaqt); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
D <- "/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex"
B <- file.path(D, "baselines", "mpaqt")
idx_file <- file.path(B, "index", "mpaqt.index.rds")
threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))
dir.create(dirname(idx_file), recursive = TRUE, showWarnings = FALSE)
cat("mpaqt", as.character(packageVersion("mpaqt")), "\n")

if (args[1] == "index") {
  t0 <- Sys.time()
  mpaqt_index(annotation = file.path(D, "ref", "isoquant_db", "annotation.gtf"),
              transcriptome = file.path(D, "ref", "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa"),
              output_file = idx_file, read_length = 150L, threads = threads,
              temp_dir = file.path(B, "index", "tmp"))
  cat("index:", format(Sys.time() - t0), "\n")
} else {
  cell <- args[2]; rep <- args[3]
  tasks <- fread(file.path(D, "tasks.tsv"))
  sr <- tasks[cell == ..cell & rep == as.integer(..rep) & protocol == "Illumina"]
  index <- mpaqt_read_index(idx_file)
  sr_dir <- file.path(B, sprintf("%s_rep%s_Illumina", cell, rep))
  sr_rds <- file.path(sr_dir, "mpaqt.short_read.rds")
  t0 <- Sys.time()
  sr_counts <- if (file.exists(sr_rds)) mpaqt_prepare_short_reads(index, rds_file = sr_rds) else
    mpaqt_prepare_short_reads(index, fastq_1 = sr$reads1, fastq_2 = sr$reads2, output_dir = sr_dir,
                              threads = threads)
  cat("short reads:", format(Sys.time() - t0), "\n")
  for (lr in c("directcDNA", "cDNA")) {
    name <- sprintf("%s_rep%s_%s", cell, rep, lr)
    out <- file.path(B, name)
    dir.create(out, showWarnings = FALSE)
    ab <- fread(file.path(D, "baselines", "lr-kallisto", name, "abundance_1.tsv"))
    counts_csv <- file.path(out, "lr_kallisto_counts.csv")
    fwrite(ab[, .(transcript_id = target_id, count = est_counts)], counts_csv)
    t0 <- Sys.time()
    lr_counts <- mpaqt_prepare_long_reads(index, bambu_counts = counts_csv, output_dir = out)
    res <- mpaqt_quant(index = index, sr_counts = sr_counts, lr_counts = lr_counts)
    mpaqt_save_result(res, output_dir = out, prefix = name)
    cat(name, "quant:", format(Sys.time() - t0), "\n")
  }
}
