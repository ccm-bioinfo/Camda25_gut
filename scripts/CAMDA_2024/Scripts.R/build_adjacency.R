# Matriz de adyacencia CUADRADA del grafo bipartito función–taxón
# Equivalente en R del pipeline de Python.

library(Matrix)

infile   <- "function_taxon_split.tsv"
out_base <- "adjacency_bipartite"

# 1) Leer y quitar UNINTEGRATED
df <- read.delim(infile, header = FALSE, col.names = c("funcion", "taxon"),
                 stringsAsFactors = FALSE)
df <- df[df$funcion != "UNINTEGRATED", ]
df <- unique(df)

# 2) Nodos: funciones (ordenadas) + taxones (ordenados)
funciones <- sort(unique(df$funcion))
taxones   <- sort(unique(df$taxon))
nodos     <- c(funciones, taxones)
N         <- length(nodos)

i <- match(df$funcion, nodos)
j <- match(df$taxon,   nodos)

# 3) Matriz dispersa simétrica binaria
A <- sparseMatrix(i = c(i, j), j = c(j, i), x = 1L,
                  dims = c(N, N), dimnames = list(nodos, nodos))
A <- as.matrix(A)            # densa para exportar con nombres
storage.mode(A) <- "integer"

# 4) Guardar CSV y TSV con nombres de fila y columna
write.csv(A, paste0(out_base, ".csv"))
write.table(A, paste0(out_base, ".tsv"), sep = "\t",
            quote = FALSE, col.names = NA)

cat(sprintf("Nodos: %d (funciones=%d, taxones=%d)\n", N, length(funciones), length(taxones)))
cat(sprintf("Aristas no dirigidas: %d\n", sum(A) %/% 2))
cat(sprintf("Simétrica: %s | Diagonal cero: %s\n",
            isSymmetric(A), sum(diag(A)) == 0))
