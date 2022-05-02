# Large Data Processing in R Workshop

* Naeem Khoshnevis 


## Summary

R is a language and environment for statistical computing and graphics, which is created by statisticians Ross Ihaka and Robert Gentleman. It was released in 1995 as open-source software. The base R is limited to a single thread on the CPU and in-memory computation. However, R has been used in numerous projects because of a well-organized community (e.g., CRAN, Rstudio, tidyverse) and wrapper packages to connect to many advanced APIs. This workshop presents some common problems in using large data in R and possible solutions. 

<p align="center" width="100%">
    <img width="60%" src="figures/png/R_and_wrappers.png">
</p>

## Computational Resources

Systems with different computational configurations are more accessible than before. Sometimes, having too many options can become overwhelming. This workshop also presents solving a simple, embarrassingly parallel problem on different computational resources; these resources include shared and distributed memory systems. 

<p align="center" width="100%">
    <img width="60%" src="figures/png/computing_systems.png">
</p>

## Problems

- [Working with large data that does not fit into memory](data_format/NYC_lyft.md)
- [Processing Single instruction multiple data problem on shared and distributed memory systems](parallel_computation/R_embarrassingly_parallel.md)

## Need help?

If you have any questions, please read the documentation.

- https://docs.rc.fas.harvard.edu/

If you cannot find an answer or you need further help, please submit a ticket.

- https://portal.rc.fas.harvard.edu/login/?next=/rcrt/submit_ticket