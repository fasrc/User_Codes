
data <- read.csv("mpi_data.csv")

ggplot(data, aes(log2(ncore), log2(wc))) + 
               geom_point() + 
               geom_smooth(method = "lm") + 
               coord_fixed(ratio = 1, xlim = NULL, ylim = NULL,
                           expand = TRUE, clip = "on")

lm(data = data, formula = log2(wc) ~ log2(ncore))