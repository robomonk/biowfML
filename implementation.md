

![alt text](https://gencore.bio.nyu.edu/wp-content/uploads/2016/03/VariantCallingWorkflow-Updated.jpg)

# Running the GATK4 Docker Containers utlizing Ray cluster and RLLib for self optimzing data data steam processing DAG.    

## Introduction 

This document outlines how to integrate the `gencorefacility/variant-calling-pipeline-gatk4` Docker image into a Directed Acyclic Graph (DAG) based workflow for variant calling. This approach is ideal for parallel processing and ensures reproducibility of each pipeline step.

**Docker Image:** `gencorefacility/variant-calling-pipeline-gatk4`

**Docker Hub Page:** [https://hub.docker.com/r/gencorefacility/variant-calling-pipeline-gatk4](https://hub.docker.com/r/gencorefacility/variant-calling-pipeline-gatk4)

**Understanding the Container's Role:**

Based on typical bioinformatics Docker image design and the nature of a complex pipeline like GATK4 variant calling, the `gencorefacility/variant-calling-pipeline-gatk4` image is likely intended as a **"tool box" container**. This means it provides a consistent environment with all the necessary tools (BWA, Samtools, GATK4) installed and configured. It's not designed to run the entire pipeline end-to-end with a single command from outside the container.

Instead, your DAG-based workflow management system will orchestrate the execution of individual pipeline steps by:

1.  **Launching a container** based on `gencorefacility/variant-calling-pipeline-gatk4` for each task.
2.  **Mounting necessary data directories** from your host system into the container.
3.  **Executing the specific command** for that pipeline step *inside* the container, referencing the mounted files.

**Prerequisites for Your Workflow:**

*   **Workflow Management System:** Choose and set up a DAG-based workflow system (e.g., Nextflow, Snakemake, Cromwell).
*   **Docker Installed:** Docker must be installed and running on your execution environment.
*   **Reference Genome:** The reference genome file used by Gencore Bio's pipeline should be available on your system and indexed for tools like BWA and GATK.
*   **Input FASTQ Files:** Your raw sequencing data (paired-end FASTQ files) should be organized and accessible.

**Conceptual Workflow Tasks and Container Commands:**

Below is a breakdown of the common steps in a GATK4 variant calling pipeline and the likely commands you would execute *within* the `gencorefacility/variant-calling-pipeline-gatk4` container for each task in your DAG.

**Important:** The paths within the container (e.g., `/path/to/input_R1.fastq.gz`) will correspond to the paths you define when mounting volumes using the `-v` flag in your `docker run` commands (orchestrated by your workflow system).

---

### Task 1: BWA Alignment

*   **Description:** Aligns sequencing reads to the reference genome using BWA.
*   **Input:** R1 FASTQ file, R2 FASTQ file, Reference Genome (indexed by BWA).
*   **Output:** Unsorted BAM file.
*   **Container Command (within the container):**

    ```bash
    bwa mem [BWA_OPTS] /path/to/reference.fa /path/to/input_R1.fastq.gz /path/to/input_R2.fastq.gz > /path/to/output.bam
    ```
    *   *Note:* Consult the Docker Hub page or any associated documentation for specific `[BWA_OPTS]` recommended by Gencore Bio.

---

### Task 2: Samtools Sort and Index

*   **Description:** Sorts the aligned BAM file and creates an index for faster access.
*   **Input:** Unsorted BAM file.
*   **Output:** Sorted BAM file, BAM index file (`.bai`).
*   **Container Commands (within the container):**

    ```bash
    samtools sort /path/to/input.bam -o /path/to/output_sorted.bam
    samtools index /path/to/output_sorted.bam /path/to/output_sorted.bam.bai
    ```

---

### Task 3: GATK Mark Duplicates

*   **Description:** Identifies and marks potential PCR duplicates.
*   **Input:** Sorted BAM file.
*   **Output:** Marked Duplicates BAM file, Metrics file.
*   **Container Command (within the container):**

    ```bash
    gatk MarkDuplicates \
      --INPUT /path/to/input_sorted.bam \
      --OUTPUT /path/to/output_marked_duplicates.bam \
      --METRICS_FILE /path/to/output_metrics.txt \
      [MARK_DUPLICATES_OPTS]
    ```
    *   *Note:* Check for any specific `[MARK_DUPLICATES_OPTS]` they might use.

---

### Task 4: GATK Base Quality Score Recalibration (BQSR - First Pass)

*   **Description:** Builds a recalibration model based on empirical data and known variants.
*   **Input:** Marked Duplicates BAM file, Known Variants VCF (e.g., dbSNP VCF).
*   **Output:** Recalibration Table.
*   **Container Command (within the container):**

    ```bash
    gatk BaseRecalibrator \
      --INPUT /path/to/input_marked_duplicates.bam \
      --REFERENCE /path/to/reference.fa \
      --known-sites /path/to/known_variants.vcf.gz \
      --OUTPUT /path/to/output_recal_table.txt \
      [BQSR_OPTS]
    ```
    *   *Note:* You will need to provide the path to a suitable known variants VCF file (often downloaded from GATK resource bundles).

---

### Task 5: GATK Apply BQSR

*   **Description:** Applies the recalibration model to adjust base quality scores in the BAM file.
*   **Input:** Marked Duplicates BAM file, Recalibration Table.
*   **Output:** Recalibrated BAM file.
*   **Container Command (within the container):**

    ```bash
    gatk ApplyBQSR \
      --INPUT /path/to/input_marked_duplicates.bam \
      --OUTPUT /path/to/output_recalibrated.bam \
      --bqsr-recal-file /path/to/input_recal_table.txt \
      [APPLY_BQSR_OPTS]
    ```

---

### Task 6: GATK HaplotypeCaller (per sample)

*   **Description:** Calls variants (SNPs and Indels) for a single sample and outputs a gVCF file.
*   **Input:** Recalibrated BAM file, Reference Genome.
*   **Output:** gVCF file.
*   **Container Command (within the container):**

    ```bash
    gatk HaplotypeCaller \
      --INPUT /path/to/input_recalibrated.bam \
      --REFERENCE /path/to/reference.fa \
      --output-mode EMIT_GVCF_ONLY \
      --OUTPUT /path/to/output.g.vcf.gz \
      [HAPLOTYPECALLER_OPTS]
    ```

---

### Task 7: GATK GenotypeGVCFs (Joint Genotyping)

*   **Description:** Combines gVCF files from multiple samples and performs joint genotyping.
*   **Input:** Multiple gVCF files (one from each sample after HaplotypeCaller).
*   **Output:** Jointly Genotyped VCF file.
*   **Container Command (within the container):**

    ```bash
    gatk GenotypeGVCFs \
      --REFERENCE /path/to/reference.fa \
      --VCF /path/to/sample1.g.vcf.gz \
      --VCF /path/to/sample2.g.vcf.gz \
      # ... add --VCF /path/to/sampleN.g.vcf.gz for each sample ...
      --OUTPUT /path/to/output_joint_genotyped.vcf.gz \
      [GENOTYPEGVCFS_OPTS]
    ```
    *   *Note:* This task runs *after* the `HaplotypeCaller` task has completed for all samples in your cohort.

---

### Task 8: Variant Filtering

*   **Description:** Applies filters to the jointly genotyped VCF to remove low-confidence calls.
*   **Input:** Jointly Genotyped VCF file.
*   **Output:** Filtered VCF file.
*   **Container Commands (within the container):** This step can use `gatk VariantFiltration` (hard filtering) or potentially `gatk VQSR` if applicable.

    *   **Example using `gatk VariantFiltration`:**
        ```bash
        gatk VariantFiltration \
          --VCF /path/to/input_joint_genotyped.vcf.gz \
          --OUTPUT /path/to/output_filtered.vcf.gz \
          --filter-expression "QD < 2.0 || FS > 60.0 || MQ < 40.0 || MQRankSum < -12.5 || ReadPosRankSum < -8.0" \
          --filter-name "HardFilter"
        ```
        *   *Note:* The specific filter expressions should align with Gencore Bio's recommendations.

---

### Task 9: Variant Annotation

*   **Description:** Adds information about the functional consequences and database entries of the identified variants.
*   **Input:** Filtered VCF file.
*   **Output:** Annotated VCF file.
*   **Container Command (within the container):** The specific command will depend on the annotation tool included in the container or used by Gencore Bio (e.g., `snpEff`, `VEP`, `gatk Funcotator`). You'll need to check if an annotation tool is present in this specific image.

---

**Implementing in Your Workflow System:**

When defining your workflow in Nextflow, Snakemake, or Cromwell, you will configure each task to use the `gencorefacility/variant-calling-pipeline-gatk4` container and execute the relevant command as shown above. The workflow system will handle the volume mounts (`-v`) to make your local input files available inside the container and to retrieve the output files generated by each task.

By using this container for each individual step, you gain the benefits of a consistent and reproducible environment, which is crucial for reliable bioinformatics analysis within a parallel processing framework. Remember to consult the Docker Hub page and any linked documentation for the most precise command-line arguments and usage details for this specific image.



