# MPI IO on the FASRC cluster

## Introduction

MPI IO is a component of the Message Passing Interface standard that enables parallel input/output operations in distributed computing applications. It allows multiple processes to collectively access shared files, optimizing disk operations through features like collective I/O and file views that reorganize data access patterns. This capability is crucial for high-performance computing applications working with large datasets across multiple nodes, as it extends parallel processing efficiency to the I/O subsystem.

For more information, see [User Docs on MPI IO](https://docs.rc.fas.harvard.edu/kb/mpi-io/)

### Contents:
* [Example1](Example1/): Program creates 3 random vectors and writes / reads them with parallel MPI-IO
