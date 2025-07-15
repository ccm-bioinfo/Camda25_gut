# Script to create a Phyloseq object from provided data

# Load necessary libraries
library(phyloseq)

# Define the data directory (make sure to define 'Data' before running this script)
# Data <- "path/to/your/data/directory/"

# Import Metadata
meta <- read.table(paste0(Data, "metadata.txt"), header = TRUE)
rownames(meta) <- meta$SampleID
meta$Diagnosis <- as.factor(meta$Diagnosis)
meta$Project <- as.factor(meta$Project)

# Import Pathways data
path <- read.csv(paste0(Data, "pathways.csv"), row.names = 1, header = TRUE)

# Import Taxonomy data
taxo <- read.table(paste0(Data, "taxonomy.txt"), row.names = 1, header = TRUE)
# Multiply and round to avoid errors
taxo2 <- round(taxo * 100, 0)

# Create the Phyloseq object
OTU <- otu_table(as.matrix(taxo2), taxa_are_rows = TRUE)
SAM <- sample_data(meta)
physeq <- merge_phyloseq(phyloseq(OTU), SAM)

# Print the created Phyloseq object
print(physeq)
