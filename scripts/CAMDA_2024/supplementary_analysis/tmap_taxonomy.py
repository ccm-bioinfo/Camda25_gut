from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd


taxonomy = pd.read_csv('../../DataSets/CAMDA/taxonomy.txt', sep = '\t', index_col = 0)
metadata = pd.read_csv('../../DataSets/CAMDA/metadata.csv')

# Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)
# Step2. Projection
lens = [Filter.MDS(components=[0, 1],random_state=100)]
projected_X = tm.filter(taxonomy.T, lens=lens)
clusterer = DBSCAN(eps=0.75, min_samples=1)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=20, overlap=0.75)

graph = tm.map(data=StandardScaler().fit_transform(X), cover=cover, clusterer=clusterer)

print(len(graph.nodes),len(graph.edges))