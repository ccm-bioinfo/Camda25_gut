# Script for Venn Diagram of Differential Taxa (DTs)

# Load necessary libraries
library(VennDiagram)
library(gplots)

# Define directories (ensure these are defined appropriately in your script)
# Data directory containing the differential taxa files
# dirDTs <- "path/to/your/DT/directory/"
# dirVenn <- "path/to/your/Venn/directory/"

# Combination contrasts
comb_contrasts <- t(combn(contrasts, 2))

# Input files
countFiles <- list.files(path = dirDTs, pattern = "_0.05.txt")

# Create a list of genes for each contrast in the given condition
list_Venn <- list()
for (count in countFiles) {
    name <- sub(paste0("-", condition, ".txt"), "", count)
    data <- read.table(paste0(dirDTs, count), 
                       header = TRUE, sep = "\t", quote = "", 
                       row.names = 1, comment.char = "")
    list_Venn[[name]] <- data
}

# Analysis
for (ud in c("", "up.", "down.")) {
    # Define contrasts
    x <- contrasts[1]
    y <- contrasts[2]
    z <- contrasts[3]
    
    # Create venn list
    vennlist <- list(
        x = rownames(list_Venn[[paste0(ud, "DT-", x)]]),
        y = rownames(list_Venn[[paste0(ud, "DT-", y)]]),
        z = rownames(list_Venn[[paste0(ud, "DT-", z)]])
    )
    names(vennlist) <- c(paste0(ud, x), paste0(ud, y), paste0(ud, z))
    
    # Create Venn diagram without plot (to get intersections)
    ventemp <- venn(vennlist, show.plot = FALSE)
    ventemp <- attributes(ventemp)$intersections
    ventemp <- t(do.call(rbind, ventemp))
    
    # Create and save Venn diagram
    venn.diagram(
        x = vennlist, 
        category.names = names(vennlist), 
        main = "Venn Diagram", 
        sub = paste0(ud, x, " vs ", ud, y, " vs ", ud, z),
        filename = paste0(dirVenn, ud, "DT_", x, "-vs-", y, "-vs-", z, ".svg"), 
        output = TRUE
    )
    
    # Save table with intersections
    write.table(
        ventemp, 
        file = paste0(dirVenn, "ListVenn-", ud, "DT_", x, "-vs-", y, "-vs-", z, ".txt"),
        row.names = FALSE, quote = FALSE, sep = "\t", col.names = TRUE
    )
}
