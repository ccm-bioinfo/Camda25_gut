#!/usr/bin/env Rscript
## -----------------------------------------------------------------------------------------------------------------------


args = commandArgs(trailingOnly=TRUE)

#if (!require(vegan)) install.packages('vegan')
#library(vegan)
if (!require(igraph)) install.packages('igraph')
library(igraph)
#library(ggplot2)
#if (!require(apcluster)) install.packages('apcluster')
#library(apcluster)
if (!require(plyr)) install.packages('plyr')
library(plyr)
if (!require(stringr)) install.packages('stringr')
library(stringr)


#getwd()
setwd("..")

#####DATOS#####

data <- paste0("./DataSets/", args[1])

if (str_sub( data , -4 , -1) == ".csv" ){
  data <- read.csv(data , row.names = 1 , header = TRUE)
} else {
  data <- read.table(data , row.names = 1, header = TRUE , sep = "" )  
}




#Normalización
for (i in 1:dim(data)[2]){
  data[,i] <- data[,i]/sum(data[,i])
}


#######ANÁLISIS DE OTUS######

data$nodos <- 0:(dim(data)[1]-1)

#Eliminación de otus según su aparición en muestras
filt <- c()
for (i in 1:dim(data)[1]) {
  
  v_i <- as.vector(data[i,1:(dim(data)[2]-1)])
  #el siguiente 1 es filtro
  if (length(v_i [ v_i > 0 ]) > 1 ) {
    filt <- c(filt, i)
  }
}

data <- data[filt,]


######CARGA DE RED Y AJUSTE A FILTRACIÓN DE OTUS######
red <- paste0("./DataSets/", args[2])
red <- read.csv(red)
red = red[,1:3]#se asume la forma del archivo de red

#Dado que se han filtrado otus, solo retendremos las aristas que se refieren a los otus conservados en nuestros datos y a correlaciones positivas
edges <- c()
for (i in 1:dim(red)[1]) {
  if (is.element(red[i,1], data$nodos) && is.element(red[i,2], data$nodos) && red[i,3] > 0  ){
    edges <- c(edges , i)
  }
}

red <- red[edges, 1:2]

#####AJUSTES PREVIOS AL ISO DE igraph######
#red <- red + 1

for (i in 1:dim(red)[1]){
  for (j in 1:dim(red)[2]){
    red[i,j] <- paste0("v_",as.character(red[i,j]))
  }
}


#data$nodos <- data$nodos + 1

for (i in 1:dim(data)[1]){
  data[i,"nodos"] <- paste0("v_" , as.character(data[i,"nodos"]))
}


########CARGA DE RED CON igraph Y ELECCION DE COMPONENTE CONEXA PRINCIPAL##### 


net_work <- graph_from_edgelist(as.matrix(red) , directed = FALSE )



##componente(s) conexa(s) principal(es)
compo_conexas <- components(net_work)
#size_compo_conexas <- compo_conexas$csize
#princ <- which(size_compo_conexas == max(size_compo_conexas))

componentes <- list()
data_0 <- list()
pertenencia <- compo_conexas$membership
for (i in 1:length(compo_conexas$size)){
  compo_i <- names(which(pertenencia == i))
  filtro_i <- c()
  
  for (j in 1:dim(data)[1]){
    if(is.element(data[i,"nodos"],compo_i)){
      filtro_i <- c(filtro_i, j)
    }
  }
  
  data_i <- data[filtro_i,]
  
  data_0[[i]] <- data_i
  compo_conexas[[i]] <- induced_subgraph(net_work, compo_i ,"auto")
}


#compo_princ <- which(pertenencia == princ )
#compo_princ <- names(compo_princ)



##nuevos datos





#print(dim(data))




degrees <- list()

for (j in 1:length(componentes)){
  degrees_j <- c()
  
  for (i in 1:dim(data_0[[j]])[1]) {
    d_i <- degree(componentes[[j]], data_0[[j]][i,"nodos"])
    degrees_j <- c(degrees_j, d_i)
  }
  data_0[[j]]$degrees <- degrees_j
}

## --------


closeness_cent <- list()

for (j in 1:length(componentes)){
  closeness_j <- c()
  
  for (i in 1:dim(data_0[[j]])[1]) {
    d_i <- closeness(componentes[[j]], data_0[[j]][i,"nodos"])
    closeness_j <- c(closeness_j, d_i)
  }
  data_0[[j]]$closeness <- closeness_j
}


betweenness_cent <- list()

for (j in 1:length(componentes)){
  betweenness_j <- c()
  
  for (i in 1:dim(data_0[[j]])[1]) {
    d_i <- betweenness(componentes[[j]], data_0[[j]][i,"nodos"])
    betweenness_j <- c(betweenness_j, d_i)
  }
  data_0[[j]]$betweenness <- betweenness_j
}


data_1 <- data.frame()
for (i in 1:length(data_0)){
  data_1 <- rbind.data.frame(data_1 , data_0[[i]])
}



data_deg <- data_1[order(data_1$degrees, decreasing = TRUE),]
data_close <- data_1[order(data_1$closeness , decreasing = TRUE),]
data_between <- data_1[order(data_1$betweenness, decreasing = TRUE),]





file <- args[3]

write.csv(data_deg , paste0("./DataSets/", file ,"_centrality_measures.csv") , row.names = TRUE)
#write.csv(data_close,paste0("./results/otus_by_centrality/",file,"_bycloseness.csv") , row.names = TRUE)
#write.csv(data_between, paste0("./results/otus_by_centrality/",file,"_bybetweenness.csv") , row.names = TRUE)

#report_1 <- args[4]

hdeg <- which(data$degrees >= quantile(data$degrees , probs = seq(0, 1, 0.33))[3])
hclose <- which(data$closeness >= quantile(data$closeness , probs = seq(0, 1, 0.33))[3])
lbetween <- which(data$betweenness <= quantile(data$betweenness , probs = seq(0, 1, 0.33))[2])

results_1 <- intersect(hdeg,hclose)
results_1 <- intersect(results_1 , lbetween)

data_report_1 <- data[results_1,]

write.csv(data_report_1 , paste0("./DataSets/Keystone_OTUs_", file , ".csv" ) , row.names = TRUE)



