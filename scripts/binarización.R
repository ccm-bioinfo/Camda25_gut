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
  for (j in 1:dim(data)[1]){
    if (data[j,i] > 0 ){
      data[j,i] <- 1
    }else{
      if (data[j,i] < 0){
        data[j,i] <- -1
      }
    }
  }
}

write.csv(data , paste0( "./DataSets/", args[1], "_01.csv"))