# To extract fastq files from SRA database using SRA toolkit:

prefetch SRR16497710
prefetch SRR16497711
prefetch SRR16497712
prefetch SRR16497713
prefetch SRR16497714
prefetch SRR16497715


fasterq-dump SRR16497710.sra
fasterq-dump SRR16497711.sra
fasterq-dump SRR16497712.sra
fasterq-dump SRR16497713.sra
fasterq-dump SRR16497714.sra
fasterq-dump SRR16497715.sra

# Renaming file for easier refernce:

mv SRR16497710.fastq control.fastq
mv SRR16497711.fastq control2.fastq
mv SRR16497712.fastq control3.fastq
mv SRR16497713.fastq silenced.fastq
mv SRR16497714.fastq silenced2.fastq
mv SRR16497715.fastq silenced3.fastq

# Checking components of all files using head: (step repeated for all six fastq files)

head control.fastq
echo "file finished"

# Counting the number of lines present in the file using wc

wc -l control.fastq

#Number of reads in silenced sample = 32974693
#Number of reads in control sample = 34036906
#They match the reads mentioned in the SRA database


# To run my RNAseq pipline bash script:
# make sure I gave permessions:
chmod 775 




