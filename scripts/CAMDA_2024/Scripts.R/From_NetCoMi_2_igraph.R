library(igraph)

# Convierte un objeto microNet (NetCoMi) a un grafo de igraph.
#   net     : objeto de netConstruct()
#   props   : (opcional) objeto de netAnalyze() para anexar atributos de nodo
#   group   : 1 o 2 (red 1 / red 2 si comparaste grupos con netCompare)
#   useAsso : TRUE  -> assoMat (conserva el SIGNO +/-)   <- recomendado
#             FALSE -> adjaMat (similitud en [0,1], la que dibuja NetCoMi)
netcomi_to_igraph <- function(net, props = NULL, group = 1, useAsso = TRUE) {
  
  mat <- if (useAsso) {
    if (group == 1) net$assoMat1 else net$assoMat2
  } else {
    if (group == 1) net$adjaMat1 else net$adjaMat2
  }
  if (is.null(mat)) stop("La matriz solicitada es NULL (¿existe la red 'group'?).")
  
  g <- graph_from_adjacency_matrix(
    as.matrix(mat),
    mode     = "undirected",
    weighted = TRUE,
    diag     = FALSE
  )
  
  # Si usamos asociaciones, separamos magnitud y signo
  if (useAsso) {
    E(g)$assoc  <- E(g)$weight        # valor con signo (correlacion)
    E(g)$sign   <- sign(E(g)$weight)  # +1 / -1
    E(g)$weight <- abs(E(g)$weight)   # peso = magnitud (para layouts/clustering)
  }
  
  # Atributos de nodo desde netAnalyze()
  if (!is.null(props)) {
    nm    <- V(g)$name
    clust <- if (group == 1) props$clustering$clust1   else props$clustering$clust2
    deg   <- if (group == 1) props$centralities$degree1 else props$centralities$degree2
    hubs  <- if (group == 1) props$hubs$hubs1          else props$hubs$hubs2
    
    if (!is.null(clust)) V(g)$cluster <- clust[nm]
    if (!is.null(deg))   V(g)$degree  <- deg[nm]
    if (!is.null(hubs))  V(g)$is_hub  <- nm %in% hubs
  }
  
  g
}


g_spring    <- netcomi_to_igraph(net_spring,    props_spring,    useAsso = TRUE)
g_spieceasi <- netcomi_to_igraph(net_spieceasi, props_spieceasi, useAsso = TRUE)
g_sparcc    <- netcomi_to_igraph(net_sparcc,    props_sparcc,    useAsso = TRUE)

g_spring                                   # resumen
sum(E(g_spring)$sign < 0)                  # nº de aristas negativas
g_spring <- delete_vertices(g_spring, degree(g_spring) == 0)  # quitar nodos aislados (opcional)
write_graph(g_spring, "red_spring.graphml", format = "graphml")  # a Gephi/Cytoscape
