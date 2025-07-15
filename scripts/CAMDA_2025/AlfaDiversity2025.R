
# Load only necessary libraries
library(phyloseq)
library(ggplot2)
library(vegan)
library(gridExtra)
library(RColorBrewer)
library(dplyr)
library(tidyr)

# Define the color mapping
color_map <- c(
  'H' = '#009E73',     # Bluish Green - Healthy
  'MD' = '#0072B2',    # Blue - Metabolic
  'NP' = '#E69F00',    # Yellow - Neoplasms
  'CD' = '#F0E442',    # Light Yellow - Circulatory
  'DD' = '#D55E00',    # Vermillion - Digestive
  'ID' = '#56B4E9',    # Sky Blue - Infectious
  'MBD' = '#CC79A7'    # Reddish Purple - Mental
)

# Update the group order to match color mapping
pc <- names(color_map)

# Modify the alpha diversity data frame to use ordered factors
alpha$Group <- factor(alpha$Group, levels = pc)

# Richness plot with new colors
ggplot(data = alpha, aes(x = Group, y = richness, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, colour = "black") +
  geom_jitter(width = 0.2, size = 2, alpha = 0.8, aes(color = Group)) +
  scale_fill_manual(values = color_map) +
  scale_color_manual(values = color_map) +
  tema +
  ylab("Richness") +
  xlab("Group") +
  theme(legend.position = "none")

# Shannon diversity plot with new colors
ggplot(data = alpha, aes(x = Group, y = shannon, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, colour = "black") +
  geom_jitter(width = 0.2, size = 2, alpha = 0.8, aes(color = Group)) +
  scale_fill_manual(values = color_map) +
  scale_color_manual(values = color_map) +
  tema +
  ylab("Shannon Index") +
  xlab("Group") +
  theme(legend.position = "none")

# NMDS plot with new colors
ggplot(scaling3, aes(x = MDS1, y = MDS2, color = Group)) +
  geom_point(size = 3, alpha = 0.9) +
  stat_ellipse(aes(group = Group), linetype = 2) +
  scale_color_manual(values = color_map) +
  tema +
  xlab("NMDS1") +
  ylab("NMDS2") +
  ggtitle(paste0("NMDS - Stress: ", round(scaling2$stress, 3))) +
  guides(color = guide_legend(override.aes = list(size = 4))) # Larger legend points



