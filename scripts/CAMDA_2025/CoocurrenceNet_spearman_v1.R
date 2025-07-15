setwd("~/CAMDA2025/Camda25_gut/DataSets/CAMDA_2025/")

data_path<-read.table("pathways.txt", header = TRUE, sep="\t",  quote= "", row.names=1, comment.char="")
head(data_path)

metadata<-read.table("metadata_corrected_final.txt", header = TRUE, sep="\t",  quote= "", row.names=NULL)
head(metadata)

path_des<-read.table("Pathways_Types_complete_new.txt", header = TRUE, sep="\t",  quote= "", row.names=NULL)
head(path_des)

#First making a merge of specifict pathway with layers of pathways

path_des_meta<-path_des
colnames(path_des_meta)

#Clear names

pathways<-sub(":.*","", rownames(data_path))
head(pathways)

data_path.1<-data_path
data_path.1$Object.ID<-pathways

head(data_path.1[,c(1,4399)], 20)

#Merge pathways with layer pathways

merge_data_path.1<-merge(data_path.1, path_des, by="Object.ID")


#### Selecting by group ####

head(metadata)

ID<-metadata[metadata$Group=="MBD","sample"]
head(ID)          

ID_path<-data_path[,ID]
rownames(ID_path)<-sub(":.*","",rownames(ID_path))

head(ID_path) 

#### Correlation network

dataMat<-ID_path

#dataMat<-dataMat[,-2116]

dataMat<-t(dataMat)
sample<-colnames(dataMat)
paths<-rownames(dataMat)

library(gplots)

Y.u=t(log(dataMat+(1e-6)))
Y.u=t(dataMat)
par(mfrow=c(2,1))
hist(as.matrix(dataMat), breaks=50, main="normalized data")
hist(as.vector(Y.u), breaks=50, main="logarithm of the normalized data")
par(mfrow=c(1,1))

corr.u=cor(t(Y.u))
dim(corr.u)
corr.s=cor(t(Y.u), method="spearman")
hist(corr.s[upper.tri(corr.s)])

#Fisher's Z transformation
n=ncol(dataMat)   
z.s=0.5*log((1+corr.s)/(1-corr.s))
summary(z.s[upper.tri(z.s)])
hist(z.s[upper.tri(z.s)])

#### similarity matrix plot ##################################################
corr.s_dist<-as.matrix(dist(corr.s[1:50,1:50], method = "euclidean" ))
corr.s_dist[is.na(corr.s_dist)]=0

corr.s[is.na(corr.s)]=0
#z.s_dist<-as.matrix(dist(z.s[1:235,1:235], method = "euclidean"))
heatmap.2(corr.s_dist, #corr.s
          col=redgreen(75),
          labRow=NA, labCol=NA, 
          trace='none', 
          hclustfun = function(corr.s_dist) hclust(corr.s_dist, method = "average"),
          xlab='tax', ylab='tax', dendrogram ="row",
          main='Similarity matrix',
          density.info='none', revC=TRUE)

thre.z=qnorm(0.9)  ## normal quanitle
adjcent.z=abs(z.s)>thre.z  ## symmetric adjacency matrix: 1: there is an edge; 0 : there is no edge

#thre.z=0.3## normal quanitle
#adjcent.z=pnorm(abs(z.s), lower.tail = FALSE)<thre.z

diag(adjcent.z)=0  ## taxons do not connect themselves in the network
rownames(adjcent.z)=rownames(corr.u)
colnames(adjcent.z)=colnames(corr.u)
sum(adjcent.z)/2
adjcent.z[is.na(adjcent.z)]=0

adj_dist<-as.matrix(dist(adjcent.z[1:100,1:100], method = "euclidean"))
heatmap.2(t(adjcent.z[1:100,1:100]),
          col=redgreen(75),
          labRow=NA, labCol=NA, 
          trace='none', dendrogram='row',
          #hclustfun = function(adj_dist) hclust(adj_dist, method = "average"),
          xlab='tax', ylab='tax',
          main='Adjacency matrix',
          density.info='none', revC=TRUE)

#write.csv(adjcent.z, "adjcent_path_completeNET.csv")
index=rowSums(adjcent.z)>0
weight.adjcent.z=adjcent.z[index,index]
weight.adjcent.z[!upper.tri(weight.adjcent.z)]<-0
corr.se<-corr.s[index,index]

weight_a=weight.adjcent.z>0
weight_b<-weight_a*corr.se
weight_b[upper.tri(weight_b)]

weight_E<-t(weight_b)[abs(t(weight_b))>0]

library(igraph)

g=graph.adjacency(weight.adjcent.z, mode="undirected", diag=FALSE)

#### continue ####
community.fastgreedy=fastgreedy.community(g)
community.fastgreedy
table(community.fastgreedy$membership)

#write.csv(data.frame(cbind(b,lab_prop_comm$membership)),"dge_clouvain.genes.csv")

hist(betweenness(g))
b <- betweenness(g, normalized=TRUE)

df.z.g=rowSums(weight.adjcent.z)
hub <- df.z.g

c <- community.fastgreedy$membership
key <- cbind(b, hub, c)
#write.csv(data.frame(key),"key_path_completeNET.csv")

V(g)$color <- "grey57"
for(i in 1:length(unique(community.fastgreedy$membership))){
  V(g)[community.fastgreedy$membership==i]$color=i
  
  if(length(which(community.fastgreedy$membership==i))<7){
    V(g)[community.fastgreedy$membership==i]$color="grey"
  }
}

V(g)$comm <- 0
for(i in 1:length(unique(community.fastgreedy$membership))){
  V(g)[community.fastgreedy$membership==i]$comm=i
  
  if(length(which(community.fastgreedy$membership==i))<7){
    V(g)[community.fastgreedy$membership==i]$comm=0
  }
}

sort(unique(V(g)$comm)) # you know number of communities and community 

pal<-colorRampPalette(c("#df65b0","#1f78b4","#33a02c","#006837","#a6cee3",
                        "#e31a1c","#d7301f","#ff7f00","#cab2d6","#6a3d9a",
                        "#eef071","#eef071","#eef071","#eef071","#b15928",
                        "#b15928","#35978f","#35978f"))(n=18)#repeat #b15928 to 14 
# and #969696 to 17, for corresponding the community number 
#with color position in pal

V(g)$color <- "grey57"
for(i in 1:length(unique(community.fastgreedy$membership))){
  V(g)[community.fastgreedy$membership==i]$color=pal[i]
  
  if(length(which(community.fastgreedy$membership==i))<7){
    V(g)[community.fastgreedy$membership==i]$color="#D9D9D9"
  }
}

pal_legend<-colorRampPalette(c("#d9d9d9","#df65b0","#1f78b4","#33a02c","#006837","#a6cee3",
                               "#e31a1c","#d7301f","#ff7f00","#cab2d6","#6a3d9a",
                               "#eef071","#b15928","#35978f"))(n=14)

node.col<-V(g)$color

V(g)[df.z.g>20]$color <- "black" # to play around with hub gene thresholds
v.label=rep("",length(V(g)))
v.label=V(g)$name  # if you want to put gene name
v.size=rep(20,length(V(g)))
# v.size[V(g)$name %in% "AP2"]=4 # if you want to change size of specific nodes
V(g)$shape <- "circle"
V(g)$hub<-hub
V(g)$name

V(g)$label<-V(g)$name


#pdf("taxon_ID_UID.pdf", useDingbats=FALSE, width = 15,height = 15)

#svg("ID_net_path.svg", width = 6, height = 6)  # Tamaño en pulgadas

plot(g, layout=layout.fruchterman.reingold(g), 
     vertex.size=v.size, 
     vertex.frame.color=node.col,
     #vertex.label = NA,
     vertex.label= ifelse(V(g)$hub>45, V(g)$label, NA),
     vertex.label.cex=0.55,edge.color="gray57", edge.width =0.1,
     rescale = FALSE, ylim=c(-14,14),xlim=c(-25,25), asp = 1
)

legend("topright", legend=paste("cluster ",sort(unique(V(g)$comm))), col = unique(pal_legend), pch = 19, title = "Community")
#dev.off()

evw<-data.frame(name=V(g)$name, cluster = V(g)$color, hub=V(g)$hub)
head(evw)

write.table(evw, file="path_hubc_CompleteNET.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")

evw_black<-evw[evw$cluster=="black",]

write.table(evw_black, file="hblack_CompleteNET.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")


head(path_des_meta)

rownames(path_des_meta)<-path_des_meta$Object.ID

select_path_des_meta<-path_des_meta[evw$name,]

colnames(select_path_des_meta)

colnames(evw)<-c("Object.ID","cluster","hub")

evw_merge<-merge(evw,select_path_des_meta, by ="Object.ID")

evw_merge$General_Pathway_Type
evw_merge$Parental.Ontology.pathway_type

write.table(evw_merge, file="path_merge_CompleteNET.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")

library(ggplot2)
library(dplyr)

# Contar frecuencias
df_plot <- evw_merge %>%
  count(General_Pathway_Type, name = "Frequency") %>%
  arrange(desc(Frequency))

# Plot

#svg("ID_pathwaytype.svg", width = 6, height = 6)  # Tamaño en pulgadas
png("pathwaytype_CompleteNET.png", width = 700, height = 500,units = "px")  # Tamaño en pulgadas

ggplot(df_plot, aes(x = factor(General_Pathway_Type, levels = rev(sort(unique(General_Pathway_Type)))), y = Frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # para que las categorías estén en el eje Y
  theme_minimal() +
  theme(
    text = element_text(size = 10),           # Tamaño general del texto
    axis.text = element_text(size = 14),      # Tamaño de texto en ejes
    axis.title = element_text(size = 10),     # Tamaño de títulos de ejes
    plot.title = element_text(size = 10, face = "bold")
  ) +
  labs(
    title = "Pathway Types Frequency",
    x = "General Pathway Type",
    y = "Count"
  )

dev.off()


# Contar frecuencias
df_plot <- evw_merge %>%
  count(Parental.Ontology.pathway_type, name = "Frequency") %>%
  arrange(desc(Frequency))

#svg("ID_pathway_parental.svg", width = 6, height = 6)  # Tamaño en pulgadas
png("pathway_parental_CompleteNET.png", width = 700, height = 500,units = "px")  # Tamaño en pulgadas


ggplot(df_plot, aes(x = factor(Parental.Ontology.pathway_type, levels = rev(sort(unique(Parental.Ontology.pathway_type)))), y = Frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # para que las categorías estén en el eje Y
  theme_minimal() +
  theme(
    text = element_text(size = 6),           # Tamaño general del texto
    axis.text = element_text(size = 14),      # Tamaño de texto en ejes
    axis.title = element_text(size = 10),     # Tamaño de títulos de ejes
    plot.title = element_text(size = 10, face = "bold")
  ) +
  labs(
    title = "Pathway Types Frequency",
    x = "Parental.Ontology.pathway_type",
    y = "Count"
  )

dev.off()


head(evw_merge)

evw_merge_hub<-evw_merge[evw_merge$cluster=="black",]

write.table(evw_merge_hub, file="path_merge_hub_CompleteNET.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")


#######

data_path$ID_path<-pathways
data_path$pathways<-rownames(data_path)

head(rownames(data_path))
rownames(data_path)<-data_path$ID_path
paths.ID<-data.frame(short=rownames(data_path), complete_path=data_path$pathways)
head(paths.ID)

rownames(paths.ID)<-paths.ID$short

head(paths.ID)




list.files()
fileslist<-list.files(pattern = "hblack")
fileslist
fileslist<-fileslist[-4]

list1<-list()

id<-c()

for(i in fileslist){
  print(i)
  namesf<-sub("_.*","", i)
  
  data = read.table(i, header = TRUE, sep="\t",  quote= "", row.names=1, comment.char="")
  rownames(data)<-data$name
  
  #paths.ID_hub_d<-paths.ID[rownames(data),]
  #rownames(paths.ID_hub_d)<-1:length(rownames(paths.ID_hub_d))
  
  #write.table(paths.ID_hub_d, file=paste("pathways_hub_", namesf, ".txt",sep=""), row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")
  
  list1[[namesf]]<-data
  id<-unique(c(id, rownames(list1[[namesf]])))
  
}
length(id)
id

tab<-data.frame(row.names = id)

for(x in names(list1)){
  print(paste("table add:", x))
  
  df=list1[[x]]
  coln<-x
  print(coln)
  
  tab[rownames(df),coln]<-df[,3]
  
}
head(tab)

tab[is.na(tab)]=0

head(tab)

write.table(tab, file="blackhubs_pathways.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")




#pathways<-sub(":.*","", rownames(data_path))
#head(pathways)

#data_path$ID_path<-pathways
#data_path$pathways<-rownames(data_path)

#head(rownames(data_path))
#rownames(data_path)<-data_path$ID_path
#paths.ID<-data.frame(short=rownames(data_path), complete_path=data_path$pathways)
#head(paths.ID)

#rownames(paths.ID)<-paths.ID$short

#head(paths.ID)

#paths.ID_hub<-paths.ID[id,]
#rownames(paths.ID_hub)<-1:length(rownames(paths.ID_hub))
#paths.ID_hub<-paths.ID_hub[,-1]

#write.table(tab, file="hubs_namespaths.txt", row.names=TRUE, col.names=NA, quote=FALSE, sep="\t")

