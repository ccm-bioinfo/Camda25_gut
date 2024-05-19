# Script to create a Phyloseq object from .biom file

# Load necessary libraries
library(phyloseq)

# Define the data directory (make sure to define 'Data' before running this script)
# Data <- "path/to/your/data/directory/"

# Import and manipulation of .biom file
biom <- import_biom(paste0(Data, "file.biom"))
biom@tax_table@.Data <- substring(biom@tax_table@.Data, 4)
colnames(biom@tax_table@.Data) <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

# Metadata
meta <- read.table(paste0(Data, "metadata.txt"), header = T)
rownames(meta) <- meta$SampleID
meta$Diagnosis <- as.factor(meta$Diagnosis)
meta$Project <- as.factor(meta$Project)
biom@sam_data <- sample_data(meta)

# Print the created Phyloseq object
print(biom)