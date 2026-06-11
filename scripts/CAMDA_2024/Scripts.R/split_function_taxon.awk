x <- readLines("function_taxon_CD.txt")
x <- x[grepl("\\|", x)]                       # quita el encabezado "x"
val <- sub('^"[0-9]+" "(.*)"$', "\\1", x)     # extrae el valor entre comillas
funcion <- sub(":.*", "", sub("\\|.*", "", val))  # código antes de ':'
taxon   <- sub("^[^|]*\\|", "", val)              # todo tras el primer '|'
write.table(data.frame(funcion, taxon),
            "salida.tsv", sep = "\t",
            row.names = FALSE, col.names = FALSE, quote = FALSE)
