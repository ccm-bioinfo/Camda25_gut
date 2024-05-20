# Script for calculating and plotting alpha and beta diversity

# Load necessary libraries
library(phyloseq)
library(ggplot2)

# Alpha Diversity
measures <- c("Observed", "Chao1", "ACE", "Shannon", "Simpson")
alpha <- plot_richness(physeq = physeq, measures = measures,
                       color = "Diagnosis", x = "Diagnosis", 
                       title = "Alpha Diversity Indices - Data Provided") +
    labs(x = "Diagnosis", y = "Alpha Diversity Measures") +
    theme_classic() +
    geom_boxplot(aes(fill = Diagnosis), alpha = 0.7)

# Print alpha diversity plot
print(alpha)

# Beta Diversity
Ord <- ordinate(physeq = physeq, method = "PCoA", distance = "bray")
beta <- plot_ordination(physeq = physeq, ordination = Ord, color = "Diagnosis") +
    theme_classic() +
    geom_point(size = 3) +
    geom_text(mapping = aes(label = Diagnosis), size = 4, vjust = 2, hjust = 1) + 
    geom_vline(xintercept = 0) +
    geom_hline(yintercept = 0) +
    labs(title = "Beta diversity (PCoA Bray-Curtis) - Data Provided")

# Print beta diversity plot
print(beta)

