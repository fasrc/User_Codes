#!/usr/bin/env Rscript
# R <https://www.r-project.org/about.html>
# Thank you to Connor Jerzak for this example
args <- commandArgs(TRUE)
i_ <- as.integer(args[1])
cat('Hello world! My first argument is', i_, '\r\n')
