# Installing Software from Pre-built Binaries

This tutorial is intended to walk you through the required steps to install software packages on the FAS-RC Cannon cluster available as pre-compiled binaries. The software apps are frequently packaged in archives with extensions, such as <code>.tar</code>, <code>.tar.gz</code>, <code>.zip</code>, etc.

## Realistic Example

The installation process is best illustrated by a specific example. Suppose, you need a tool to extract files packaged as <code>.rar</code> archives. Although, readily available on Windows and MAC computers, such a tool is not available on the cluster out of the box. One such tool that works from the command-line can be downloaded from [RARLAB](https://www.rarlab.com/). Here are the exact steps to make this work:

* ### Download the package from the web

First we need to locate the package we need. Going to the [Downloads](https://www.rarlab.com/download.htm) page at the RARLAB's website, we locate the package we need, <code>RAR 6.12 for Linux x64</code>. We get the package's URL (right-click and then "Copy Link Address") and download it with <code>wget</code> in a selected directory for this purpose.

```bash
$ mkdir -p $HOME/Software 
$ cd $HOME/Software 
$ wget https://www.rarlab.com/rar/rarlinux-x64-612.tar.gz
```
* ### Extract the archive

The above commands will result in the file <code>rarlinux-x64-612.tar.gz</code> which can be unpacked with the <code>tar</code> command, e.g.,

```bash
$ tar xvfz rarlinux-x64-612.tar.gz
rar/
rar/unrar
rar/acknow.txt
rar/whatsnew.txt
rar/order.htm
rar/readme.txt
rar/rar.txt
rar/makefile
rar/default.sfx
rar/rar
rar/rarfiles.lst
rar/license.txt
```

This results in the <code>rar/</code> directory with the above contents. The next step is to examine its contents and to configure the binary for use.

* ### Examine the unpacked files

At this stage the binaries are usually ready for use. Of you go to the unpacked directory and list its contents, you will see something like the below:

```bash
$ cd rar/
$ ls -lh
total 1.4M
-rw-r--r-- 1 pkrastev rc_admin 2.7K May  4 14:22 acknow.txt
-rwxr-xr-x 1 pkrastev rc_admin 199K May  4 14:22 default.sfx
-rw-r--r-- 1 pkrastev rc_admin 6.6K May  4 14:22 license.txt
-rw-r--r-- 1 pkrastev rc_admin  428 May  4 14:22 makefile
-rw-r--r-- 1 pkrastev rc_admin 3.3K May  4 14:22 order.htm
-rwxr-xr-x 1 pkrastev rc_admin 624K May  4 14:22 rar
-rw-r--r-- 1 pkrastev rc_admin 1.2K May  4 14:22 rarfiles.lst
-rw-r--r-- 1 pkrastev rc_admin 105K May  4 14:22 rar.txt
-rw-r--r-- 1 pkrastev rc_admin  692 May  4 14:22 readme.txt
-rwxr-xr-x 1 pkrastev rc_admin 348K May  4 14:22 unrar
-rw-r--r-- 1 pkrastev rc_admin  28K May  4 14:22 whatsnew.txt
```
The executable files of interest are <code>rar</code> and <code>unrar</code>. In case these are dynamically linked executables, it is useful to check if all required libraries are present for them to work. This is done by the <code>ldd</code> command, e.g.,

```bash
$ ldd rar
        linux-vdso.so.1 =>  (0x00007ffe46f7f000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00002afdb1e68000)
        libm.so.6 => /lib64/libm.so.6 (0x00002afdb2170000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00002afdb2472000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00002afdb2688000)
        libc.so.6 => /lib64/libc.so.6 (0x00002afdb28a4000)
        /lib64/ld-linux-x86-64.so.2 (0x00002afdb1c44000)
```
No missing libraries in this case, so we are good to go! At this stage the binary is ready to be used from its working directory. You may check, for instance, the available options to be used with the executable (binary), e.g.,

```bash
$ ./rar

RAR 6.12   Copyright (c) 1993-2022 Alexander Roshal   4 May 2022
Trial version             Type 'rar -?' for help

Usage:     rar <command> -<switch 1> -<switch N> <archive> <files...>
               <@listfiles...> <path_to_extract\>

<Commands>
  a             Add files to archive
  c             Add archive comment
  ch            Change archive parameters
  cw            Write archive comment to file
  d             Delete files from archive
  e             Extract files without archived paths
  f             Freshen files in archive
  i[par]=<str>  Find string in archives
  k             Lock archive
  l[t[a],b]     List archive contents [technical[all], bare]
  m[f]          Move to archive [files only]
  p             Print file to stdout
  r             Repair archive
  rc            Reconstruct missing volumes
  rn            Rename archived files
  rr[N]         Add data recovery record
  rv[N]         Create recovery volumes
  s[name|-]     Convert archive to or from SFX
  t             Test archive files
  u             Update files in archive
  v[t[a],b]     Verbosely list archive contents [technical[all],bare]
  x             Extract files with full path

<Switches>
  -             Stop switches scanning
  @[+]          Disable [enable] file lists
  ad[1,2]       Alternate destination path
  ag[format]    Generate archive name using the current date
  ai            Ignore file attributes
  ap<path>      Set path inside archive
  as            Synchronize archive contents
  c-            Disable comments show
  cfg-          Disable read configuration
  cl            Convert names to lower case
  cu            Convert names to upper case
  df            Delete files after archiving
  dh            Open shared files
  ds            Disable name sort for solid archive
  dw            Wipe files after archiving
  e[+]<attr>    Set file exclude and include attributes
  ed            Do not add empty directories
  ep            Exclude paths from names
  ep1           Exclude base directory from names
  ep3           Expand paths to full including the drive letter
  ep4<path>     Exclude the path prefix from names
  f             Freshen files
  hp[password]  Encrypt both file data and headers
  ht[b|c]       Select hash type [BLAKE2,CRC32] for file checksum
  id[c,d,n,p,q] Display or disable messages
  ierr          Send all messages to stderr
  ilog[name]    Log errors to file
  inul          Disable all messages
  isnd[-]       Control notification sounds
  iver          Display the version number
  k             Lock archive
  kb            Keep broken extracted files
  log[f][=name] Write names to log file
  m<0..5>       Set compression level (0-store...3-default...5-maximal)
  ma[4|5]       Specify a version of archiving format
  mc<par>       Set advanced compression parameters
  md<n>[k,m,g]  Dictionary size in KB, MB or GB
  me[par]       Set encryption parameters
  ms[ext;ext]   Specify file types to store
  mt<threads>   Set the number of threads
  n<file>       Additionally filter included files
  n@            Read additional filter masks from stdin
  n@<list>      Read additional filter masks from list file
  o[+|-]        Set the overwrite mode
  oh            Save hard links as the link instead of the file
  oi[0-4][:min] Save identical files as references
  ol[a]         Process symbolic links as the link [absolute paths]
  op<path>      Set the output path for extracted files
  or            Rename files automatically
  ow            Save or restore file owner and group
  p[password]   Set password
  qo[-|+]       Add quick open information [none|force]
  r             Recurse subdirectories
  r-            Disable recursion
  r0            Recurse subdirectories for wildcard names only
  rr[N]         Add data recovery record
  rv[N]         Create recovery volumes
  s[<N>,v[-],e] Create solid archive
  s-            Disable solid archiving
  sc<chr>[obj]  Specify the character set
  sfx[name]     Create SFX archive
  si[name]      Read data from standard input (stdin)
  sl<size>      Process files with size less than specified
  sm<size>      Process files with size more than specified
  t             Test files after archiving
  ta[mcao]<d>   Process files modified after <d> YYYYMMDDHHMMSS date
  tb[mcao]<d>   Process files modified before <d> YYYYMMDDHHMMSS date
  tk            Keep original archive time
  tl            Set archive time to latest file
  tn[mcao]<t>   Process files newer than <t> time
  to[mcao]<t>   Process files older than <t> time
  ts[m,c,a,p]   Save or restore time (modification, creation, access, preserve)
  u             Update files
  v<size>[k,b]  Create volumes with size=<size>*1000 [*1024, *1]
  ver[n]        File version control
  vn            Use the old style volume naming scheme
  vp            Pause before each volume
  w<path>       Assign work directory
  x<file>       Exclude specified file
  x@            Read file names to exclude from stdin
  x@<list>      Exclude files listed in specified list file
  y             Assume Yes on all queries
  z[file]       Read archive comment from file
```

and also similarly for the <code>unrar</code> command, e.g.,

```bash
$ ./unrar
...
<omitted output>
```
If you intend to use the binaries from the directory where they are unpacked, this would be the last step. 

* ### Add the executable to the PATH environment variable

You can also make the executables "visible" from other locations by adding the executables' path to the <code>PATH</code> environment variable by:

```bash
export PATH=$HOME/Software/rar:$PATH
```
The above command can be also added to your <code>.bashrc</code> file in your home directory to make it available upon logging on to the cluster.

* ### Using the binary

Finally, you are ready to use the install! 

To open/extract a RAR file in the current working directory, just use the following command with <code>unrar e</code> option:

```bash
$ unrar e my_file.rar
```
To open/extract a RAR file in a specific path or destination directory, just use the <code>unrar e</code> option, it will extract all the files in the specified destination directory:

```bash
unrar e my_file.rar /path/to/my/directory/
```

To open/extract a RAR file with its original directory structure, just issue the below command with <code>unrar x</code> option. It will extract according to their folder structure see below the output of the command:

```bash
$ unrar x example.rar
```
All files will be unpacked in the <code>example/</code> directory.


