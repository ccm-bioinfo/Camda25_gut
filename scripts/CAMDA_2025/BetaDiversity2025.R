##########################################################################
##########################################################################
##CAMDA2024

tema=theme(axis.text.x = element_text(color="black",size=12, angle=0,hjust=0.5,vjust=1.5, family = "sans" ),
           #axis.text.x = element_text(color="black",size=12, angle=90,hjust=0.5,vjust=1.5),
           axis.text.y = element_text(color="black",size=12, vjust = 1.5, family = "sans"),
           axis.title = element_text(color="black",size=12, face = "bold", family = "sans"),
           axis.title.x.bottom = element_blank(),
           panel.border =element_rect(color = "black", fill = NA),#element_blank(),
           strip.text.x = element_text(size=12, color="black",face="bold", family = "sans"),
           strip.text.y = element_text(size=12, color="black",face="bold", family = "sans"),
           strip.placement = "outside", strip.background = element_rect(fill = "white"), 
           panel.background = element_rect(fill = "white",colour = "white",size = 0.8, linetype = "solid"),
           panel.grid.major.y = element_blank(),panel.grid.minor.x = element_blank(),
           legend.position = "right", legend.text = element_text(color = "black",size=12, family = "sans"), 
           legend.direction = "vertical", legend.title = element_text(color = "black",size=12, face = "bold", family = "sans"),
           legend.key.size = unit(0.4,"cm"))

my_pal=colorRampPalette(c("#024AAC","#1DACA8","#10B62F","#E2E41C","#F48F06","#F2252C","#D140B7", "grey80","dimgrey"))
pc=c("H", "MD", "NP", "CD", "DD", "ID", "MBD")

##paqueterias
library(phyloseq)
library(tidyverse)
library(vegan)
library(RColorBrewer)
library(patchwork)

#######
## Generate OTU table
setwd("~/Camda2025/")
## Generate OTU table
data.dir=("~/Camda2025/")
main.dir=getwd()

datos<- read.delim("taxa_corrected.txt", sep = "\t", header = TRUE)
metadatos <- read.delim(file=paste(getwd(),paste("metadata_final","txt",sep = "."),sep = "/"))
colnames(metadatos)

rownames(metadatos)=metadatos$sample
metadatos <- metadatos[metadatos$Group != "delete", ]
#################
#tax <- read.delim("taxa_corrected.txt",sep = "\t", header = TRUE)
rownames(datos)=datos$X.sample.id
datos <- datos[, -1]
#datos <- t(datos)
#datos=datos[rownames(metadatos),]
####################


# Transpone los datos para que las especies estén en las filas
abundancia <- as.matrix(datos[, -1])
rownames(abundancia) <- datos$Species

metadata_df <- as.data.frame(metadatos)
rownames(metadata_df) <- metadata_df$sample
# Crea el objeto phyloseq
physeq <- phyloseq(otu_table(abundancia, taxa_are_rows = TRUE), sample_data(metadata_df))

#plot_richness(physeq = physeq, measures = c("cohort","Shannon_Entropy_on_Functions")) 

# Calcula la diversidad alfa (por ejemplo, riqueza de especies)
alfa_diversity <- estimate_richness(physeq)

# Visualiza la diversidad alfa
print(alfa_diversity)

#beta diversity
meta_ord <- ordinate(physeq = physeq, method = "NMDS", distance = "bray")
plot_ordination(physeq = physeq, ordination = meta_ord, color = "Group")+
  theme_minimal()

##Extraer las coordenadas del objeto de ordenación
ord_df <- as.data.frame(meta_ord$points)
ord_df$SampleID <- rownames(ord_df)


ggplot(ord_df, aes(x = MDS1, y = MDS2)) +
  geom_point() +
  geom_text(aes(label = SampleID), size = 3, vjust = -1) +
  theme_minimal()
######################################## filtrar los outliers
outliers <- c("train_55", "train_474")
physeq_filtered <- prune_samples(!(sample_names(physeq) %in% outliers), physeq)

meta_ord_filtered <- ordinate(physeq_filtered, method = "NMDS", distance = "bray")

plot_ordination(physeq_filtered, meta_ord_filtered, color = "Group") +
  theme_minimal()+tema

# Abrir el dispositivo SVG
png("betadiversity.png", width = 8.5, height = 11, units = "in", res = 600, type = "cairo")
print(
  plot_ordination(physeq_filtered, meta_ord_filtered, color = "Group") +
    theme_minimal() + tema
)
dev.off()

######ggplot2
ord_data <- plot_ordination(physeq_filtered, meta_ord_filtered, justDF = TRUE)

png("betadiversity.png", width = 8.5, height = 8.5, units = "in", res = 600, type = "cairo")
print(
  ggplot(data = ord_data, aes(x = NMDS1, y = NMDS2, 
                            colour = Group)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_point(size = 3) +
  scale_colour_manual(values = c("gray90", "gray80", "gray50", "gray30",
                                   "firebrick", "darkcyan", "goldenrod")) +
  scale_alpha_continuous(range = c(0.4, 1)) +
  scale_shape_manual(values = c(17, 16, 15, 18, 19, 20, 21)) +
  labs(x = "NMDS1", y = "NMDS2") +
  theme_minimal() + tema  # tu tema personalizado
)
dev.off()

###SUBSET
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y MBD
ord_data_subset_1 <- ord_data[ord_data$Group %in% c("H", "MBD"), ]

ggplot(data = ord_data_subset_1, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  theme_minimal() +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "MBD" = "darkorange")) +
  scale_shape_manual(values = c(17, 16)) +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

###SUBSET2
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y ID
ord_data_subset_2 <- ord_data[ord_data$Group %in% c("H", "ID"), ]

ggplot(data = ord_data_subset_2, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "ID" = "darkorange")) +
  scale_shape_manual(values = c(19, 20)) +
  theme_minimal() +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

###SUBSET3
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y NP
ord_data_subset_3 <- ord_data[ord_data$Group %in% c("H", "NP"), ]

ggplot(data = ord_data_subset_3, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  theme_minimal() +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "NP" = "darkorange")) +
  scale_shape_manual(values = c(17, 16)) +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

###SUBSET4
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y DD
ord_data_subset_4 <- ord_data[ord_data$Group %in% c("H", "DD"), ]

ggplot(data = ord_data_subset_4, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  theme_minimal() +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "DD" = "darkorange")) +
  scale_shape_manual(values = c(17, 16)) +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

###SUBSET5
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y MD
ord_data_subset_5 <- ord_data[ord_data$Group %in% c("H", "MD"), ]

ggplot(data = ord_data_subset_5, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  theme_minimal() +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "MD" = "darkorange")) +
  scale_shape_manual(values = c(17, 16)) +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

###SUBSET6
unique(ord_data$Group)
# Por ejemplo, solo los grupos H y MD
ord_data_subset_6 <- ord_data[ord_data$Group %in% c("H", "CD"), ]

ggplot(data = ord_data_subset_6, aes(x = NMDS1, y = NMDS2, color = Group)) +
  geom_point(size = 5) +
  theme_minimal() +
  labs(x = "NMDS1", y = "NMDS2") +
  scale_color_manual(values = c("H" = "steelblue", "CD" = "darkorange")) +
  scale_shape_manual(values = c(17, 16)) +
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))


#####abudance
species_df <- psmelt(physeq)
species_df$OTU[species_df$Abundance < 10] <- "Species < 10.0 abund"
species_df$OTU <- as.factor(species_df$OTU)
species_df_color <- colorRampPalette(brewer.pal(8,"Dark2")) (length(levels(species_df$OTU)))  
plot_species <- ggplot(data=species_df, aes(x=Sample, y=Abundance, fill=OTU))+ 
  geom_bar(aes(), stat="identity", position="stack") + scale_fill_manual(values = species_df_color)
plot_species 
