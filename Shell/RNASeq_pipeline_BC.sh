#!/bin/bash

SECONDS=0

# change working directory
cd /home/u0149935/RNASeq_pipeline/


# STEP 1: Run fastqc
fastqc data/control.fastq -o data/
fastqc data/control2.fastq -o data/
fastqc data/control3.fastq -o data/
fastqc data/silenced.fastq -o data/
fastqc data/silenced2.fastq -o data/
fastqc data/silenced3.fastq -o data/



# STEP 2: Mapping with the reference genome using HISAT2
# get the genome indices
# wget https://genome-idx.s3.amazonaws.com/hisat/grch38_genome.tar.gz

# run alignment
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/control.fastq | samtools sort -o data/control.bam
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/control2.fastq | samtools sort -o data/control2.bam
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/control3.fastq | samtools sort -o data/control3.bam
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/silenced.fastq | samtools sort -o data/silenced.bam
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/silenced2.fastq | samtools sort -o data/silenced2.bam
hisat2 -q --rna-strandness R -x genome/grch38/genome -U data/silenced3.fastq | samtools sort -o data/silenced3.bam
echo "HISAT2 finished running! ^.^"

# To take a look at the bam file in the terminal:
# samtools view -h control.bam less



# STEP 3: Run featureCounts - Quantification
# get gtf
# wget http://ftp.ensembl.org/pub/release-106/gtf/homo_sapiens/Homo_sapiens.GRCh38.106.gtf.gz

featureCounts -S 2 -a annotation/Homo_sapiens.GRCh38.106.gtf -o data/test_counts \
data/control.bam data/control2.bam data/control3.bam data/silenced.bam data/silenced2.bam data/silenced3.bam 

cat data/test_counts.summary

# Create the final tsv file that contains the gene count information:
cut -f1,7- data/test_counts | grep -v '^#' > data/sample_counts.tsv
head data/test_genedata.tsv

echo "featureCounts finished running! ^.^"

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
