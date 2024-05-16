#!/usr/bin/env Rscript
## -----------------------------------------------------------------------------------------------------------------------


args = commandArgs(trailingOnly=TRUE)

#if (!require(vegan)) install.packages('vegan')
#library(vegan)
if (!require(igraph)) install.packages('igraph')
library(igraph)
library(ggplot2)
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



######CARGA DE RED Y AJUSTE A FILTRACIÓN DE OTUS######
red <- paste0("./DataSets/networks/", args[2])
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

#head(data)
########CARGA DE RED CON igraph Y ELECCION DE COMPONENTE CONEXA PRINCIPAL##### 


net_work <- graph_from_edgelist(as.matrix(red) , directed = FALSE )
#print(V(net_work))
#plot(net_work)
#net_work

label <- c()
nodos <- data$nodos
#print(nodos)

for (i in V(net_work)){
  nodo_v <- V(net_work)[i]
  #print(names(v))
  #print(names(nodo_v))
  nodo_v <- which(nodos == names(nodo_v))
  #print(nodo_v)
  nodo_v <- data[nodo_v , 1:(dim(data)[2]-1)]
  if (length(nodo_v[ nodo_v > 0 ]) > 1){
    label <- c(label , 1)
  }
  else{
    label <- c(label , 0)
  }
}


V(net_work)$label <- label
V(net_work)$size <- 3

grafica <- plot(net_work)
ggsave("prueba.png", grafica , device = "png")
#Eliminación de otus según su aparición en muestras
