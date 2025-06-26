

{
  library(devtools)
  library(phyloseq)
  library(GGally)
  library(microbiome)
  library(genefilter)
  library(huge)
  library(pulsar)
  library(MASS)
  library(seqtime)
  library(remotes)
  library(SpiecEasi)
  library(igraph)
  library(qgraph)
  library(ggnet)
  library(RColorBrewer)
  library(grid)
  library(gridExtra)
  library(network)
  library(sna)
  library(ggplot2)
  library(ggnetwork)
  library(tidyr)
  library(Hmisc)
  library(rgexf)
  library(Rmisc)
  library(FSA)
  library(intergraph)
  library(Matrix)
}

#Choose all setting to plots
tema=theme(axis.text.x = element_text(color="black",size=16, angle=0,hjust=0.5,vjust=1.5, family = "sans" ),
           #axis.text.x = element_text(color="black",size=12, angle=90,hjust=0.5,vjust=1.5),
           axis.text.y = element_text(color="black",size=16, vjust = 1.5, family = "sans"),
           axis.title = element_text(color="black",size=16, face = "bold", family = "sans"),
           axis.title.x.bottom = element_blank(),
           panel.border =element_rect(color = "black", fill = NA),#element_blank(),
           strip.text.x = element_text(size=12, color="black",face="bold", family = "sans"),
           strip.text.y = element_text(size=12, color="black",face="bold", family = "sans"),
           strip.placement = "outside", strip.background = element_rect(fill = "white"), 
           panel.background = element_rect(fill = "white",colour = "white",size = 0.8, linetype = "solid"),
           panel.grid.major.y = element_blank(),panel.grid.minor.x = element_blank(),
           legend.position = "right", legend.text = element_text(color = "black",size=14, family = "sans"), 
           legend.direction = "vertical", legend.title = element_text(color = "black",size=12, face = "bold", family = "sans"),
           legend.key.size = unit(0.4,"cm"))

my_pal=colorRampPalette(c("#024AAC","#1DACA8","#10B62F","#E2E41C","#F48F06","#F2252C","#D140B7", "grey80","dimgrey"))
pc=c("H", "MD", "NP", "CD", "DD", "ID", "MBD")


## Generate OTU table
setwd("~/Camda2025/")
## Generate OTU table
data.dir=("~/Camda2025/")
main.dir=getwd()
met=read.delim(file=paste(getwd(),paste("metadata_final","txt",sep = "."),sep = "/"))
rownames(met) <- met$sample
rownames(met)=met$sample
met <- met[met$Group != "delete", ]
#################
tax <- read.delim("taxa_corrected.txt",sep = "\t", header = TRUE)
rownames(tax)=tax$X.sample.id
tax <- tax[, -1]
tax <- t(tax)
#tax <- round(tax)
#tax <- tax[rowSums(tax) > 0, ]
tax= tax[rownames(met),]


###############################
set.seed(23171341)
ric <- rowSums(tax > 0)
#all data
otu.s=t(rrarefy(t(tax), sample = min(rowSums(tax))))
#Checklist
colSums(otu.s)
#shan=diversity(t(otu.s),index = "shannon")
shann <- diversity(tax, index = "shannon")
########################## ALFA DIVERSIDAD #####################################

#Generate plot data
alpha=data.frame(richness=ric, shannon=shann)
alpha=alpha[met$sample,]
alpha=cbind(alpha, met[,c("category","sample","Group")])
alpha$Group <- factor(alpha$Group, levels = pc)

#########
mean.r=summarySE(data = alpha, measurevar = "richness", groupvars = "Group")
# Ordenar niveles del factor por los valores de N (de mayor a menor)
mean.r$Group=factor(mean.r$Group, level = pc)

mean.s=summarySE(data = alpha, measurevar = "shannon", groupvars = "Group")
mean.s$Group=factor(mean.s$Group, level = pc)


############# RICHNESS_1

ggplot(data=mean.r, aes(x=Group, y=richness))+
  geom_bar(stat="identity", fill=c("gray100", "gray80", "gray50", "gray30",
                                   "firebrick",    # rojo ladrillo
                                   "darkcyan",     # cian oscuro
                                   "goldenrod"),    # dorado opaco
  colour="black",size=1)+
  geom_errorbar(aes(ymin=richness-sd, ymax=richness+sd),width=0.2,position=position_dodge(1.5))+
  #geom_dotplot(data=alpha,binaxis='y', stackdir='center', stackratio=2, dotsize=0.8, aes(fill=treatment))+
  scale_fill_manual(values = my_pal(9)[4:8])+tema

############# RICHNESS_2
png("alfadiversity.png", width = 8.5, height = 11, units = "in", res = 600, type = "cairo")
print(
ggplot(data = alpha, aes(x = Group, y = richness, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, colour = "black") +
  geom_jitter(width = 0.2, size = 2, alpha = 0.8, aes(color = Group)) +
  scale_fill_manual(values = c("gray90", "gray80", "gray50", "gray30",
                               "firebrick", "darkcyan", "goldenrod")) +
  scale_color_manual(values = c("gray90", "gray80", "gray50", "gray30",
                                "firebrick", "darkcyan", "goldenrod")) +
  tema +  # tu tema predefinido
  ylab("richness") +
  xlab("Grupo") +
  theme(legend.position = "none")+tema
)
dev.off()

#####SHANNON
ggplot(data=mean.s, aes(x=Group, y=shannon))+
  geom_bar(stat="identity",fill=c("gray90", "gray80", "gray50", "gray30",
                                  "firebrick",    # rojo ladrillo
                                  "darkcyan",     # cian oscuro
                                  "goldenrod"),    # dorado opaco
  colour="black",size=1)+
  geom_errorbar(aes(ymin=shannon-sd, ymax=shannon+sd),width=0.2,position=position_dodge(1.5))+
  #geom_dotplot(data=alpha,binaxis='y', stackdir='center', stackratio=2, dotsize=0.8, aes(fill=treatment))+
  scale_fill_manual(values = my_pal(9)[1:4])+tema

###############shannon
png("shannondiversity.png", width = 8.5, height = 11, units = "in", res = 600, type = "cairo")
print(
ggplot(data = alpha, aes(x = Group, y = shannon, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, colour = "black") +  # sin puntos atípicos por separado
  geom_jitter(width = 0.2, size = 2, alpha = 0.8, aes(color = Group)) +  # puntos individuales
  scale_fill_manual(values = c("gray90", "gray80", "gray50", "gray30",
                               "firebrick", "darkcyan", "goldenrod")) +
  scale_color_manual(values = c("gray90", "gray80", "gray50", "gray30",
                                "firebrick", "darkcyan", "goldenrod")) +
  tema +  # tu tema personalizado si ya lo definiste
  ylab("Shannon Index") +
  xlab("Grupo") +
  theme(legend.position = "none")+tema
)
dev.off()


###########BETA DIVERSIDAD
tax <- tax[rowSums(tax) > 0, ]


#otu.n=as.data.frame(tax)/colSums(tax)*100 #relative abundance
otu.n=as.data.frame(t(t(otu.s)/colSums(otu.s)*100)) #relative abundance
otu.n=log10(otu.n+1)
otu.n=sqrt(otu.n)
###check
otu.n <- otu.n[!duplicated(otu.n), ]
#calculate dimensions
scaling=vegdist(t(otu.n), method = "bray", binary = T, na.rm = FALSE) #calculate distance
#scaling2=isoMDS(scaling) ; scaling2$stress #create NMDS 
scaling2=metaMDS(otu.n, distance = "bray") ; scaling2$stress
#scaling2=monoMDS(scaling); scaling2$stress
scaling3=data.frame(scaling2$species) #select cordenates
scaling3=cbind(scaling3,alpha)
####
scaling3 <- scaling3[rownames(scaling3) != "train_474", ]
scaling3 <- scaling3[rownames(scaling3) != "train_56", ]


####
d <- as.matrix(otu.n)
zero_pairs <- which(d == 0 & row(d) != col(d), arr.ind = TRUE)
print(zero_pairs)
otu.n <- otu.n[-unique(c(zero_pairs)), ]

ggplot(data=scaling3, aes(x=MDS1, y=MDS2, colour=Group, shape=Group))+
  geom_hline(yintercept = 0, linetype=2)+
  geom_vline(xintercept = 0,linetype=2)+
  geom_point(size=5)+
  scale_colour_manual(values = my_pal(31))+
  scale_alpha_continuous(range=c(0.4,1))+
  scale_shape_manual(values=c(15:19, 22, 23))+
  theme(legend.key.size = unit(0.3,"cm"), legend.text = element_text(size = 14, face = "plain"))

ggplot(scaling3, aes(x = MDS1, y = MDS2, color = Group)) +
  geom_point(size = 3, alpha = 0.9) +
  stat_ellipse(aes(group = Group), linetype = 2) +
  scale_color_manual(values = c("gray90", "gray80", "gray50", "gray30",
                                "firebrick", "darkcyan", "goldenrod")) +
  tema +
  xlab("NMDS1") +
  ylab("NMDS2") +
  ggtitle(paste0("NMDS - Stress: ", round(scaling2$stress, 3)))



###########
# 1) Carga la tabla de abundancias (evitando que R interprete '#' como comentario)
otu <- read.table(
  file        = file.choose(),      # selecciona taxonomy healthy.tsv
  header      = TRUE,
  sep         = "\t",
  row.names   = 1,
  comment.char= ""
)


# 2) Transponer: ahora filas = muestras, columnas = ASVs
otu_t <- t(otu)

# 3) Carga tu metadata completa
meta <- read.table(
  file              = file.choose(),  # selecciona metadata.txt
  header            = TRUE,
  sep               = "\t",
  stringsAsFactors  = FALSE
)

meta <- meta[meta$Group != "delete", ]

# 4) Filtrar solo las muestras 'healthy'
met <- subset(meta, Group == "H")
#met <- subset(meta, Group == "MD")
rownames(met) <- met$sample

otu_t=otu_t[rownames(met),]
  
# 7) Cálculo de distancias Bray–Curtis
dist_bc <- vegdist(otu_t, method = "bray")


# 8) PCoA (cmdscale)
pcoa <- cmdscale(dist_bc, eig = TRUE, k = 2)

# 9) Gráfico coloreado por cohorte
cols <- as.numeric(factor(group_vector))

plot(
  pcoa$points, 
  col   = "blue", 
  pch   = 19,
  xlab  = "PCoA 1", 
  ylab  = "PCoA 2",
  main  = "Beta-diversidad (Bray–Curtis) - H"
)

legend(
  "topright",
  #legend = levels(factor(group_vector)),
  #col    = seq_along(levels(factor(group_vector))),
  pch    = 19,
  cex    = 0.7
)


