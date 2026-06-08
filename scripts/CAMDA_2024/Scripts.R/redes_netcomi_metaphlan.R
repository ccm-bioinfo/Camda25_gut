# =====================================================================
# Inferencia de redes de coabundancia con NetCoMi
# Datos de entrada: abundancia relativa de MetaPhlAn (taxa x muestras)
# Objetivo: UNA red por método -> SPIEC-EASI, SparCC y SPRING
# Grupo de Biologia Cuantitativa - UAQ
# =====================================================================

# ---------------------------------------------------------------------
# 0. Instalacion (ejecutar UNA sola vez; descomentar lo necesario)
# ---------------------------------------------------------------------
# install.packages(c("remotes", "BiocManager"))
# remotes::install_github("zdk123/SpiecEasi")            # SPIEC-EASI
# remotes::install_github("GraceYoon/SPRING")            # SPRING
# remotes::install_github(
#   "stefpeschel/NetCoMi",
#   dependencies = c("Depends", "Imports", "LinkingTo"),
#   repos = c("https://cloud.r-project.org/", BiocManager::repositories())
# )

library(NetCoMi)
library(SpiecEasi)
library(SPRING)

set.seed(123)   # reproducibilidad

# ---------------------------------------------------------------------
# 1. Cargar la tabla y orientarla como NetCoMi la espera
#    MetaPhlAn entrega:  taxa en FILAS, muestras en COLUMNAS.
#    NetCoMi espera:     muestras en FILAS, taxa en COLUMNAS  -> transponer.
# ---------------------------------------------------------------------

#ruta <- "tax_CD"   # <-- ajusta la ruta (y el separador)

#tab <- read.delim(
 # ruta,
#  header       = TRUE,
 # row.names    = 1,      # 1a columna = nombres de taxa
  #check.names  = FALSE,  # conserva los SRR... tal cual
  #comment.char = "#"     # ignora encabezados tipo #mpa_vXXX de MetaPhlAn
#)

# Forzamos matriz numerica
tab<-tax_CD

tab <- as.matrix(tab)
mode(tab) <- "numeric"
cat("Original (taxa x muestras):", dim(tab), "\n")

# Transponemos: filas = muestras, columnas = taxa
otu <- t(tab)
cat("Tras transponer (muestras x taxa):", dim(otu), "\n")

# ---------------------------------------------------------------------
# 2. Filtrado de taxa (RECOMENDADO)
#    - Criterio principal: PREVALENCIA  (presente >0 en >= prev_min muestras)
#    - Criterio ligero:    ABUNDANCIA   (media >= abund_min, en unidades % de MetaPhlAn)
#    Razon: con ~1115 taxa la red es ingobernable y las taxa casi
#    siempre en cero rompen los metodos basados en log-ratios.
# ---------------------------------------------------------------------

prev_min  <- 0.20   # presente en >= 20% de las muestras
abund_min <- 0.01   # abundancia relativa media >= 0.01 %

n_muestras  <- nrow(otu)
prevalencia <- colSums(otu > 0) / n_muestras
abund_media <- colMeans(otu)

keep     <- prevalencia >= prev_min & abund_media >= abund_min
otu_filt <- otu[, keep, drop = FALSE]
cat("Taxa retenidas:", ncol(otu_filt), "de", ncol(otu), "\n")

# (OPCIONAL) Red muy legible para clase: quedarte con las N mas variables
# topN <- 50
# v    <- apply(otu_filt, 2, var)
# otu_filt <- otu_filt[, order(v, decreasing = TRUE)[seq_len(min(topN, ncol(otu_filt)))]]

# =====================================================================
# 3. Construccion de redes  ---  UNA POR METODO
#    Clave con ABUNDANCIA RELATIVA: dejamos que cada metodo haga su
#    propia transformacion interna  ->  normMethod = "none".
# =====================================================================

# ---- 3a. SPRING  (el mas apropiado para datos composicionales) ------
#   mclr (CLR modificado) tolera ceros y proporciones.
#   La esparsificacion la hace SPRING via StARS -> sparsMethod = "none".
net_spring <- netConstruct(
  data        = otu_filt,
  dataType    = "counts",
  measure     = "spring",
  measurePar  = list(nlambda = 20, rep.num = 20, Rmethod = "approx", ncores = 1),
  normMethod  = "none",
  zeroMethod  = "none",
  sparsMethod = "none",
  dissFunc    = "signed",
  verbose     = 2,
  seed        = 123
)

# ---- 3b. SPIEC-EASI  (Meinshausen-Buhlmann) -------------------------
#   Disenado para conteos; con abundancia relativa funciona, pero el
#   supuesto de conteos se viola -> interpretar con cautela.
#   StARS interna -> sparsMethod = "none".
net_spieceasi <- netConstruct(
  data        = otu_filt,
  dataType    = "counts",
  measure     = "spieceasi",
  measurePar  = list(
    method        = "mb",
    nlambda       = 20,
    pulsar.params = list(rep.num = 20, ncores = 1)
  ),
  normMethod  = "none",
  zeroMethod  = "none",
  sparsMethod = "none",
  dissFunc    = "signed",
  verbose     = 2,
  seed        = 123
)

# ---- 3c. SparCC  (composicional clasico) ----------------------------
#   Esparsificamos con bootstrap para obtener p-valores (mas riguroso
#   que un umbral fijo). nboot = 100 por velocidad; sube a 1000 para
#   resultados definitivos.
net_sparcc <- netConstruct(
  data        = otu_filt,
  dataType    = "counts",
  measure     = "sparcc",
  measurePar  = list(iter = 20, inner_iter = 10, th = 0.1),
  normMethod  = "none",
  zeroMethod  = "none",
  sparsMethod = "bootstrap",
  nboot       = 100,
  alpha       = 0.05,
  adjust      = "adaptBH",
  dissFunc    = "signed",
  cores       = 4,
  verbose     = 2,
  seed        = 123
)

# Guarda los objetos: son caros de calcular
saveRDS(net_spring,    "net_spring.rds")
saveRDS(net_spieceasi, "net_spieceasi.rds")
saveRDS(net_sparcc,    "net_sparcc.rds")

# =====================================================================
# 4. Analisis de propiedades + visualizacion
# =====================================================================

analizar_y_graficar <- function(net, titulo) {
  props <- netAnalyze(
    net,
    centrLCC    = TRUE,                  # centralidades sobre la componente conexa mayor
    clustMethod = "cluster_fast_greedy", # modulos / "guilds"
    hubPar      = "eigenvector",         # criterio para hubs
    normDeg     = FALSE
  )

  plot(
    props,
    nodeColor      = "cluster",
    nodeSize       = "eigenvector",
    nodeSizeSpread = 3,
    cexNodes       = 1.2,
    cexLabels      = 0.6,
    labelScale     = FALSE,
    hubBorderCol   = "black",
    title1         = titulo,
    showTitle      = TRUE,
    cexTitle       = 1.4
  )
  invisible(props)
}

props_spring    <- analizar_y_graficar(net_spring,    "SPRING")
props_spieceasi <- analizar_y_graficar(net_spieceasi, "SPIEC-EASI")
props_sparcc    <- analizar_y_graficar(net_sparcc,    "SparCC")

# Resumen numerico (grado, modularidad, hubs, etc.)
summary(props_spring)
summary(props_spieceasi)
summary(props_sparcc)

# =====================================================================
# NOTAS
# ---------------------------------------------------------------------
# * Costo computacional: con cientos de muestras y taxa, SPRING/SPIEC-EASI
#   (StARS) y sobre todo SparCC con bootstrap pueden tardar bastante.
#   Para probar el flujo, baja rep.num (p. ej. 5) y nboot (p. ej. 20);
#   sube a rep.num = 20-50 y nboot = 1000 para resultados finales.
# * Si mas adelante quieres COMPARAR dos grupos (p. ej. condiciones,
#   o LOSO), usa netConstruct(data, data2, group=...) + netCompare().
# * dissFunc = "signed" conserva el signo de las asociaciones; las
#   aristas negativas se dibujan en otro color por defecto.
# =====================================================================
