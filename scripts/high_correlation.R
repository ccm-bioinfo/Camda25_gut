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

#tabla de abundancias
data <- paste0("./DataSets/", args[1])
data <- read.csv(data , row.names = 1 , header = TRUE , fill = TRUE)


#red
red <- paste0(args[2])
red <- read.csv(red)
red = red[,1:3]

#lista de otus a tomar en cuenta
lista <- paste0(".DataSets/" , args[3])
lista <- which(is.element(lista, row.names(data)))



#se genera una etiqueta que concuerde con las de la red
data$nodos <- 0:(dim(data)[1]-1)


#la red se restringe a los nodos de la lista
filtro <- c()

for (i in 1:dim(red)[1]){
  if ((is.element(red[i , "taxon1"], data$nodos[lista]  )  | is.element(red[i , "taxon2"], data$nodos[lista]  )) & red[i, 3] > 0){
    filtro <- c(filtro , i)
  }
}


red <- red[filtro , 1:2]

#reajuste de etiquetas para mejor funcionamiento de igraph

for (i in 1:dim(red)[1]){
  for (j in 1:dim(red)[2]){
    red[i,j] <- paste0("v_",as.character(red[i,j]))
  }
}


for (i in 1:dim(data)[1]){
  data[i,"nodos"] <- paste0("v_" , as.character(data[i,"nodos"]))
}

#Carga de red y búsqueda de la componente principal
net_work <- graph_from_edgelist(as.matrix(red) , directed = FALSE )



##componente(s) conexa(s) principal(es)
compo_conexas <- components(net_work)
size_compo_conexas <- compo_conexas$csize
princ <- which(size_compo_conexas == max(size_compo_conexas))
pertenencia <- compo_conexas$membership
compo_princ <- which(pertenencia == princ )
compo_princ <- names(compo_princ)


#
data <- data[ which(is.element(data$nodos , compo_princ)) , ]

write.csv(data , paste0( substr(args[3] , -4 , -1) , "_correlation.csv"))

