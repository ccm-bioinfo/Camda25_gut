library(phyloseq)
library(tidyverse)
library(vegan)


datos<- read_csv("/files/camda2024/gut/DataSets/taxonomy.csv", col_names = TRUE)
metadatos <- read_csv("/files/camda2024/gut/DataSets/metadata.csv", col_names = TRUE)
 colnames(metadatos)
 # Instala las bibliotecas si aún no las tienes instaladas
# install.packages("phyloseq")
# install.packages("tidyverse")
# install.packages("vegan")
 
 # Carga las bibliotecas necesarias
 library(phyloseq)
 library(tidyverse)
 library(vegan)
 library("RColorBrewer")
 library("patchwork")

 
 # Lee los datos de abundancia de especies
 datos <- read_csv("/files/camda2024/gut/DataSets/taxonomy.csv")
 
 # Lee los metadatos
 metadatos <- read_csv("/files/camda2024/gut/DataSets/metadata.csv")
 
 # Transpone los datos para que las especies estén en las filas
 abundancia <- as.matrix(datos[, -1])
 rownames(abundancia) <- datos$Species
 

 metadata_df <- as.data.frame(metadatos)
 rownames(metadata_df) <- metadata_df$SampleID
 # Crea el objeto phyloseq
 physeq <- phyloseq(otu_table(abundancia, taxa_are_rows = TRUE), sample_data(metadata_df))
 
 #plot_richness(physeq = physeq, measures = c("Observed","Chao1","Shannon")) 
 
 # Calcula la diversidad alfa (por ejemplo, riqueza de especies)
 #alfa_diversity <- estimate_richness(physeq)
 
 # Visualiza la diversidad alfa
 #print(alfa_diversity)
 
 #beta diversity
 meta_ord <- ordinate(physeq = physeq, method = "NMDS", distance = "bray")
 plot_ordination(physeq = physeq, ordination = meta_ord, color = "Diagnosis")

 species_df <- psmelt(physeq)
 species_df$OTU[species_df$Abundance < 10] <- "Species < 10.0 abund"
 species_df$OTU <- as.factor(species_df$OTU)
 species_df_color <- colorRampPalette(brewer.pal(8,"Dark2")) (length(levels(species_df$OTU)))  
 plot_species <- ggplot(data=species_df, aes(x=Sample, y=Abundance, fill=OTU))+ 
   geom_bar(aes(), stat="identity", position="stack") + scale_fill_manual(values = species_df_color)
plot_species 
 