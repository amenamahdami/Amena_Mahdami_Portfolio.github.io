# script to perform differential gene expression analysis using DESeq2 package
setwd("c:/Users/u0149935/Documents/DESeq2_airway")

# load libraries
BiocManager::install('DESeq2')
library(DESeq2)
library(tidyverse)
library(airway)


#### Step 1: preparing count data ####
# read in counts data
counts_data <- read.csv('counts_data.csv')
head(counts_data)

# read in sample info
colData <- read.csv('sample_info.csv')

# making sure the row names in colData matches to column names in counts_data
all(colnames(counts_data) %in% rownames(colData))
#^ all the columns present in counts matrix are present as rows in sample information
# are they in the same order?
all(colnames(counts_data) == rownames(colData))


#### Step 2: construct a DESeqDataSet object ####
dds <- DESeqDataSetFromMatrix(countData = counts_data,
                              colData = colData,
                              design = ~ dexamethasone)
dds

#^ design factor is in what thing you want to compare your samples? this case it's treats VS untreated
# pre-filtering: removing rows with low gene counts
#to reduce the size of the dds dataset object and speedup computation
#This step is recommended but not required
# keeping rows that have at least 10 reads total
keep <- rowSums(counts(dds)) >= 10
#then you use the logical value 'keep' to subsit our data object 'dds
dds <- dds[keep,]
dds

# set the factor level
#To tell DESeq to compare treated with the untreated, if we don't explicitly mention the refrence level it will just aphabitically choose and use that as the refrence
dds$dexamethasone <- relevel(dds$dexamethasone, ref = "untreated")
dds$dexamethasone

# NOTE: collapse technical replicates 
# never collapse biological replicates

#### Step 3: Run DESeq ####
dds <- DESeq(dds)
res <- results(dds)
res
# log2 fold change is calculated for in the design factor 'dexamethasone' b/w treated & untreated
# whatever value we see in the FC are all in the 'treated', bc it's compared to the'untreated'
# baseMean: the avrage of the normalized counts taken over all the samples 
# log2FoldChange: is the FC of each gene in the treated condition when compared with the untreated
# log2FoldChange: positive values= upregulated genes, negative values= downregulated genes
# lfcSE: the standard error estimates for the log2FoldChange
# stat: are the wald test values for each gene
# pvalue: pvalue of the statistic for thus gene
# padj: the p adjusted value is the corrected p values for multiple testing
# padj: important to avoid detection of false positive DE genes 



#### Explore Results ####
summary(res)

# change p-value < 0.1 
res0.01 <- results(dds, alpha = 0.01)
summary(res0.01)

# contrasts
# to find information about the comparasion made under the resultsNames()
resultsNames(dds)

# But what if there are multiple levels in another dataset??
# e.g.: treated_4hrs, treated_8hrs, untreated

results(dds, contrast = c("dexamethasone", "treated_4hrs", "untreated"))


#### Visualize ####
# MA plot: 
#^is a scatter plot of log to FC VS the mean of normalized counts 
plotMA(res)

# Volcano plot:
library("EnhancedVolcano")

EnhancedVolcano(res,
                lab = rownames(res),
                x = 'log2FoldChange',
                y = 'pvalue',
                pCutoff = 10e-12,
                FCcutoff = 1.5,
                cutoffLineType = 'twodash',
                cutoffLineWidth = 0.8,
                pointSize = 4.0,
                labSize = 6.0,
                colAlpha = 1,
                legendLabels=c('Not sig.','Log (base 2) FC','p-value',
                               'p-value & Log (base 2) FC'),
                legendPosition = 'right',
                legendLabSize = 16,
                legendIconSize = 5.0)
