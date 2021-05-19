# Git and Github for Non-developers
Written by: Naeem Khoshnevis   
Research Software Engineer FASRC   
nkhoshnevis@g.harvard.edu   
Last update: May 18, 2021
----------------

Git is an open-source distributed version control system. It was built by Linus Torvalds to manage to develop and maintain different versions of the Linux kernel, which is one of the most significant open-source projects (read more [here](https://git-scm.com/)). Discussing Git's advantages and disadvantages is beyond the scope of this post. It suffices to mention that Git has become one of the most used version control systems among software engineers and researchers because of its tiny footprint, lightning-fast performance, and cheap local branching. Although there are numerous great tutorials to learn Git, we believe there is a gap between fundamentals and applications, specifically for non-developers. This post aims to provide enough of both sides to make sure that the users feel comfortable using Git and comprehend each command's internal functionality.  

Almost all materials of this tutorial come from [Pro Git](https://git-scm.com/book/en/v2) book by Scott Chacon and Ben Straub. We strongly recommend reading this book for more in-depth details.     

<br/>
<div align=center>
<img width="300" src="project_git/figures/png/pro_git.png"/></img>
</div>
<br/>

There are two main concepts in learning Git that one needs to internalize them to use Git fluently. These concepts include:
- Understanding of how to read Git history through a directed acyclic graph
- Understanding what one can do on this graph  

Please note that understanding **what you can do** is much more important than **how you can do it**. After internalizing these concepts, the rest is the matter of looking up a command to do the task.  

## The big picture

Git has an intelligent and efficient approach to manage different versions (more on this later). Each version is built **based** on the previous version, and these versions are connected through a directed acyclic graph (DAG). Git guarantees that each file is stored once, and no modification can be hidden from Git's radar through an effective mechanism. The command-line interface is the recommended approach in using Git. Because it is available on your local computer and HPC systems, you can use terminal (on macOS and Linux) or Gitbash (on Windows) to work with Git. 

DAG is a blueprint to find different versions. However, rather than space, it shows the versions at different timestamps. So for us, it is a time machine.  If we understand how to read this blueprint and what we can do with it, we will find out how to do it. The following picture is a very simple DAG that shows two working history. 

<br/>
<div align=center>
<img width="200" src="project_git/figures/png/ab_graph.png"/></img>
</div>
<br/>

It reads **B is built upon A**, **B is based on A**, or **A is the parent of B**. From this graph, we understand that we had some files or/and folders at some point, which is A, and we or someone else decided to take a snapshot of those files and keep it. Later on, we or someone else worked on those files by adding, removing, or modifying them and decided to take another snapshot and keep it, which is B. 


<div align=left style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>Taking snapshot: </strong> Storing whatever is visible to the user at that timestamp. Similar to taking pictures. This is called making a <strong>commit</strong> in Git. Each commit is a node on the commit history tree.
</div>

Let's take a look at another DAG (aka working history, aka commit history). 

<br/>
<div align=center>
<img width="450" src="project_git/figures/png/ab_graph-02.png"/></img>
</div>
<br/>

In this commit history, after B, the user(s) decided to work on two different versions. If you recall from the previous commit history, the graphs directions are towards paranets or the previous versions. This commit history is like a tree and A is the root. The branches grow with time (and, of course, based on the hard work of researchers and developers). However, unlike trees, we cannot go from the root to the top of the branch. 

So if that is the case, which we cannot traverse from the root to the top of a branch, how can we navigate the work history as a user? And how Git keeps track of different versions under the hood? The answer is the **branch name**. All branches have a name, and we can jump on them, knowing their name. 

<br/>
<div align=center>
<img width="450" src="project_git/figures/png/ab_graph-03.png"/></img>
</div>
<br/>

Internally a branch name keeps the address of the last node of the branch. So you either need to know the branch name or the address of the node. Now you might ask what address? How to generate one? Can one address point to two different locations? The answers to these questions are directly related to the internal data structures that Git uses to manage any version controlling project efficiently. Git uses three types of data structures. However, before going through them, let's review some jargon in computer science.

<div align=left style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>Data Structure: </strong> A data structure is a data organization, management, and storage format that enables efficient access and modification.
</div>
</br>
<div align=left style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>Pointer: </strong> A pointer is an object in that stores a memory address.
</div>
</br>
<div align=left style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>Hash: </strong>  A hash function is any function that can be used to map data of arbitrary size to fixed-size values, mostly hexadecimals (e.g., bb765a2)[1].
</div>
</br>
<div align=left style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>SHA-1: </strong>  Secure Hash Algorithm 1. Gets input and returns a hexadecimal number, 40 digits long.
(e.g., ab8b09066b79a239bc0d2c2321f22fc0a60e2ad4)[2].
</div>

In the next section, we will review the git data structures. 

## Git data strucure

We use data structure in a lose term. You can consider them as a data container. These data strucures are:

- Commit
- Tree (is just a list of files) 
- Blob (Binary large object)

Each commit in Git is done by someone, at a specific time, with a specific message, and most importantly, a particular set of files and folders. Commit data structure stores this information in a commit object, and store it on the disk. It generates a SHA-1 value for this object (to be able to retrieve it later from the disk) and add it to the commit history or our famous DAG. So each node on the commit history has a 40 digits hexadecimal, this is called commit pointer because it stores the address of the commit object. Sometimes, for brevity, they use the first 7 or 8 characters, whichever provides unique values. The following picture illustrates a commit pointer and a commit object.

<br/>
<div align=center>
<img width="800" src="project_git/figures/png/ab_graph-04.png"/></img>
</div>
<br/>

As you can see from the figure, the commit pointer finds the commit object (stored on the disk), and the commit object has all information about that commit. Other than the author and the commit message, there are two other fields: **tree** and **parent**. Parent stores the commit pointer of the parent. In this example, it is pointing to B. It is used internally by Git to carry out different tasks. On the other hand, the tree stores a pointer to a tree object (which is just a list of files.) A "tree" data structure has the following fields for each content:

- object permission
- object type
- object pointer
- object name

The object pointer can retrieve the actual data from the disk, and the whole tree, which simply is a list of files, can create those files with the given names. This approach guarantees that you store the actual data once, no matter how many different times you change the file name. 

Finally, the object pointer points us to the actual data, a binary large object (blob). Through this process, if you can navigate any node on the commit history, you can retrieve all files and folders at that timestamp and commit. So far, we know git is storing our data efficiently and compactly to retrieve it later if we need them, but where are those files located? This is an interesting question and is the topic of the next section. 

## .git folder

A folder that has a `.git` folder in it is a Git repository. All the mentioned data and objects and much more information are located inside this folder. It is a hidden folder, and you may probably have not seen it yet. The following figure shows a Git repository and its folders. There are many other folders which are beyond the scop of this tutorial. Feel free to take a look at them in your local repository. All mentioned objects are located in `objects` folder. We will discuss the role of other files and folders later in this tutorial. 

<br/>
<div align=center>
<img width="500" src="project_git/figures/png/git_folder.png"/></img>
</div>
<br/>

You should have a good understanding of commit history and where data is located up to this point. Please note that we still do not know **how to do it**; we only know **what it is** and **what we can do**.

## Git commit history

_TBD_

## Git commands

_TBD_

## Scenarios

_TBD_

## References

1 - https://en.wikipedia.org/wiki/Hash_function  
2 - https://en.wikipedia.org/wiki/SHA-1  