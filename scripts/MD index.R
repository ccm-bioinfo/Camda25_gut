
# Indice metagenómico de diversidad alpha
# Goretty Mendoza
# 17 de mayo 2024
# este script realiza boxplots y busca diferencias para el índice MD entre
# grupos sanos y enfermos a partir de una base de datos que contiene los
# valores MD. El artículo para este índice se puede consultar con este 
# DOI: 10.1093/femsec/fiae019


# estableciendo ruta
setwd("/Users/goretty/R/RLadies24")
indir = "/Users/goretty/R/RLadies24/input/"
outdir = "/Users/goretty/R/RLadies24/results/"
pacman::p_load(ggplot2, dplyr, FSA)

#leyendo los datos
diagnosis <- read.delim("/Users/goretty/R/RLadies24/input/Diagnosis_ID.csv", sep = ",")
data <- read.delim("/Users/goretty/R/RLadies24/input/MD_results_sinOrgan.csv", sep = ",")

#uniendo mediante una columna renombrada ambas bd
data$SampleID <- data$SAMPLE
df <- merge(data, diagnosis, by="SampleID", all=FALSE)

#convirtiendo a factor
df$Diagnosis <- as.factor(df$Diagnosis)

# graficando los cuatro grupos
ggplot(df, aes(x = Diagnosis, y = MDI)) +
  geom_boxplot(aes(fill = Diagnosis), alpha = 0.7) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.6) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 3, fill = "white") +
  scale_fill_manual(values = c("skyblue", "salmon", "orange3", "lightgreen")) +
  labs(x = "Diagnosis", y = "MD index", title = "Metagenomic Alpha-Diversity Index", fill = "Diagnosis") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.position = "none"
  )


# Anova
anova <- aov(MDI ~ Diagnosis, data= df)
summary(anova)

# Probar normalidad en los residuos del anova 
hist(anova$residuals)
shapiro.test(anova$residuals)

# ya que no se cumple el supuesto de normalidad usaremos una prueba de
# Kruskal-Wallis y posteriormente una prueba de Dunn
kruskal.test(MDI ~ Diagnosis, data = df)
dunnTest(MDI ~ Diagnosis, data = df, method = "bonferroni")

# así encontramos que el grupo Obese es significativamente distinto a
# los otros tres grupos


# Ahora comparando solo entre grupos sanos vs enfermos
df2 <- df %>%
       mutate(Group = case_when(
       Diagnosis == "Healthy" ~ "Healthy",
       TRUE ~ "Unhealthy"
       ))

str(df2)
#convirtiendo a factor
df2$Group <- as.factor(df2$Group)

# graficando sanos vs enfermos
ggplot(df2, aes(x = Group, y = MDI)) +
  geom_boxplot(aes(fill = Group), alpha = 0.7) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.6) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 3, fill = "white") +
  scale_fill_manual(values = c("salmon", "orange3")) +
  labs(x = "Diagnosis", y = "MD index", title = "Metagenomic Alpha-Diversity Index", fill = "Diagnosis") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.position = "none"
  )


# Anova
anova <- aov(MDI ~ Group, data= df2)
summary(anova)

# Probar normalidad en los residuos del anova 
hist(anova$residuals)
shapiro.test(anova$residuals)

# ya que no se cumple el supuesto de normalidad usaremos una prueba de
# Kruskal-Wallis 
kruskal.test(MDI ~ Group, data = df2)

# no encontramos diferencias estadísticamente significativas entre el
# grupo sano vs el enfermo



#