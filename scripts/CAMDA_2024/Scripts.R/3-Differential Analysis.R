# Script for Differential Taxa Analysis
# For Differential Functions Analysis it is similar 

# Load necessary libraries
library(edgeR)
library(EnhancedVolcano)

# Define directories (ensure these are defined appropriately in your script)
# Data directory containing the results
# dirDTs <- "path/to/your/output/directory/"

# Rename column names with Diagnosis category
column_names <- colnames(taxo2)
new_column_names <- as.character(meta$Diagnosis[match(column_names, meta$SampleID)])
colnames(taxo2) <- ifelse(is.na(new_column_names), column_names, new_column_names)

# Copy the table object for further processing
counts <- taxo2

# Filtering step
counts <- counts[rowSums(cpm(counts) >= 1) >= 3, ]
filtering_dim <- dim(counts)

# Groups for the DGEList object
grp <- colnames(counts)

# Create DGEList object
dge <- DGEList(counts = counts, group = grp)

# Normalization
dge <- calcNormFactors(dge)

# Estimate the dispersion
dge <- estimateCommonDisp(dge)
dge <- estimateGLMTagwiseDisp(dge)

# Design of experiments
design <- model.matrix(~ 0 + group, data = dge$samples)
colnames(design) <- levels(dge$samples$group)

# Define contrasts
my_contrasts <- makeContrasts(
    CDvsHE = CD - Healthy,
    OBvsHE = Obese - Healthy,
    UCvsHE = UC - Healthy,
    levels = design
)

# Fit the model
fit <- glmQLFit(dge, design)

# Analysis for each contrast
for (contrast in colnames(my_contrasts)) {
    # Genewise Negative Binomial GLM
    qlf <- glmQLFTest(fit, contrast = my_contrasts[, contrast])
    
    # Table of the Top Differentially Expressed Taxa
    tabtop <- topTags(qlf, n = Inf)$table
    
    # Filter taxa by FDR
    significant_taxa <- rownames(tabtop)[tabtop$FDR <= 0.05]
    
    # Save table of significant taxa
    name <- paste0("DT-", contrast, "-FDR_0.05")
    write.table(tabtop[significant_taxa, ], file = paste0(dirDTs, name, ".txt"),
                row.names = TRUE, col.names = NA, quote = FALSE, sep = "\t")
    
    # Plot and save volcano plot
    plotV <- EnhancedVolcano(
        tabtop[tabtop$FDR <= 0.05, ],
        lab = significant_taxa,
        x = "logFC",
        y = "PValue",
        title = paste0("Volcano plot of ", name),
        colAlpha = 1
    )
}
