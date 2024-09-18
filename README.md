
## Computational Skills
- **R**: Bioconductor, DESeq2, ggplot2
- **Linux**: Shell scripting, FastQC, Trimmomatic, SAMtools, HISAT2, featureCounts
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Projects

### **1. NGS Reads Processing Pipeline for Human Breast Cancer Cells**

**Tools:** Bash Script, FastQC, HISAT2, featureCounts

**Project Overview:**  
This project processes RNA-seq data from the study by S Ishihara et al. ([PMID: 35428832](https://pubmed.ncbi.nlm.nih.gov/35428832/) , [GEO: GSE186211](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186211)) to explore the role of GPR81 in breast cancer.

**Workflow:**
1. **Data Retrieval**: Extract FASTQ files from the SRA database using the SRA toolkit.
2. **Quality Control**: Run FastQC to assess the quality of reads.
3. **Read Mapping**: Map reads to the genome using HISAT2.
4. **Quantification**: Quantify reads using featureCounts.

**Results:**
- [RNA-seq counts matrix](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/Shell/sample_counts.tsv)
- [Test counts summary](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/Shell/test_counts.summary)

**Scripts:**
- [Terminal commands](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/Shell/terminal%20commands.sh)
- [Bash Script: RNA-seq Pipeline](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/Shell/RNASeq_pipeline_BC.sh)

---

### **2. Differential Gene Expression Analysis (DGEA) for Human Primary Airway Smooth Muscle Cells**

**Tools:** R, DESeq2

**Project Overview:**  
This project analyzes RNA-seq data from BE Himes et al. ([PMID: 24926665](https://pubmed.ncbi.nlm.nih.gov/24926665/) , [GEO: GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778)) to study the effects of Dexamethasone on airway smooth muscle cells.

**Workflow:**
1. **Data Retrieval**: Obtain RNA-seq counts matrix using a Bioconductor package.
2. **Data Preparation**: Read and format the data.
3. **DESeq2 Analysis**: Create a DESeq dataset object and run DESeq2 to identify differentially expressed genes.
4. **Visualization**: Explore results and generate volcano plot.

**Results:**
- Differential gene expression analysis
- ![Differential Gene Expression Volcano Plot](assets/img/Volcano_plot_airway.png)

**Scripts:**
- [Getting Data](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/R/getData_airway.r)
- [Running Analysis](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/R/DESeq2_workflow_airway.R)

---

### **3. Heart Disease Prediction**

**Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn

**Project Overview:**  
This project uses a Kaggle dataset to predict heart disease incidence. It involves data wrangling, exploration, and machine learning.

**Workflow:**
1. **Data Wrangling**: Clean and prepare the dataset.
2. **Exploration**: Perform exploratory data analysis.
3. **Machine Learning**: Apply machine learning algorithms to predict heart disease risk.
4. **Evaluation**: Evaluate model performance using various metrics.

**Script:**
- [Heart Disease Prediction Python Script](https://github.com/amenamahdami/Amena_Mahdami_Portfolio.github.io/blob/main/Python/ML_Project_Heart_Disease_Prediction.ipynb)

---

### Resources, References & Training:
- [HISAT2 Manual](https://daehwankimlab.github.io/hisat2/manual/)
- [featureCounts](https://rnnh.github.io/bioinfo-notebook/docs/featureCounts.html)
- [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/)
- [Linux/Unix Command Cheat Sheet](https://rumorscity.com/2014/08/16/6-best-linuxunix-command-cheat-sheet/)
- [Bioconductor](https://www.bioconductor.org/)
- [DESeq2 package](https://bioc.ism.ac.jp/packages/2.14/bioc/vignettes/DESeq2/inst/doc/beginner.pdf)
- [DESEQ2 R Tutorial](https://lashlock.github.io/compbio/R_presentation.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [scikit-learn Machine Learning in Python](https://scikit-learn.org/stable/)
- [scikit-learn Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
- [Introduction to Bioinformatics and Computational Biology course](https://liulab-dfci.github.io/bioinfo-combio/)
- [Google Data Analytics Certificate](https://coursera.org/share/0dd196ce17876b5d71ccc0c4695b738f) 

---
