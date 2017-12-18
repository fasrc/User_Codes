(* Compute PI in via parallel Monte-Carlo method *)
tstart = AbsoluteTime[];
Print [ "Parallel calculation of PI via Monte-Carlo method." ];

nproc = 8;
LaunchKernels[nproc];

Print[ " Number of kernels: ", nproc];

n = 10^8;
m = n / nproc;
Print [ " Total number of hits: ", n ];
Print [ " Number of hits per core: ", m ];

acceptpoint[j_] := Total[Table[ 1 - Floor[ Sqrt[ (Random[])^2 + (Random[])^2 ] ], {i,1,m} ] ];
DistributeDefinitions[n,m,acceptpoint];
t1 = AbsoluteTime[];
hits = ParallelTable[acceptpoint[j], {j,1,nproc} ];
t2 = AbsoluteTime[];
tt = t2 - t1;
hits = Total[ hits ];
pi = hits / ( nproc * m ) * 4.0;
Print [ " Computed PI = ", pi ];
Print [ " Time in parallel calculation: ", tt ] ;
tend = AbsoluteTime[];
ttotal = tend - tstart;
Print [ " Total time: ", ttotal  ];
Quit [ ]
