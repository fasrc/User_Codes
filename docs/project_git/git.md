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


<div align=center style="background-color: #fcfcef; color: #061870; padding: 15px; font-size: large;">
<strong>Taking snapshot: </strong> Storing whatever is visible to the user at that timestamp. Similar to taking pictures.
</div>

