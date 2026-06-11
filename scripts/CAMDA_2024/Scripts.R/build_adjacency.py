#!/usr/bin/env python3
"""
Construye la matriz de adyacencia CUADRADA del grafo bipartito función–taxón.

  - Lee function_taxon_split.tsv (2 columnas: función <tab> taxón).
  - Elimina las filas UNINTEGRATED.
  - Nodos = funciones (ordenadas) + taxones (ordenados)  ->  N x N.
  - Matriz simétrica binaria: A[i,j] = 1 si existe la arista función–taxón.
    Bloques diagonales (func-func, taxón-taxón) = 0 por ser grafo bipartito.
  - Guarda CSV y TSV, ambos con nombres de fila y columna.

Uso:  python3 build_adjacency.py function_taxon_split.tsv salida_base
"""
import sys
import numpy as np
import pandas as pd

infile   = sys.argv[1] if len(sys.argv) > 1 else "function_taxon_split.tsv"
out_base = sys.argv[2] if len(sys.argv) > 2 else "adjacency_matrix"

# 1) Leer y quitar UNINTEGRATED
df = pd.read_csv(infile, sep="\t", header=None, names=["funcion", "taxon"])
df = df[df["funcion"] != "UNINTEGRATED"].drop_duplicates()

# 2) Construir lista de nodos: funciones primero, luego taxones
funciones = sorted(df["funcion"].unique())
taxones   = sorted(df["taxon"].unique())
nodos     = funciones + taxones
idx       = {n: i for i, n in enumerate(nodos)}
N         = len(nodos)

# 3) Matriz simétrica binaria
A = np.zeros((N, N), dtype=np.int8)
for f, t in zip(df["funcion"], df["taxon"]):
    i, j = idx[f], idx[t]
    A[i, j] = 1
    A[j, i] = 1

mat = pd.DataFrame(A, index=nodos, columns=nodos)

# 4) Guardar CSV y TSV con nombres
mat.to_csv(f"{out_base}.csv", sep=",")
mat.to_csv(f"{out_base}.tsv", sep="\t")

print(f"Nodos: {N}  (funciones={len(funciones)}, taxones={len(taxones)})")
print(f"Aristas (no dirigidas): {int(A.sum() // 2)}")
print(f"Densidad: {A.sum() / (N*N):.4f}")
print(f"Simétrica: {np.array_equal(A, A.T)}  |  Diagonal cero: {A.diagonal().sum()==0}")
