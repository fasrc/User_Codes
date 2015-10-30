(***********************************************************)
(* Program: pi_monte_carlo.m                               *)
(*          Calculation of PI via Monte-Carlo simulation   *)
(*                                                         *)
(* Run:     math -script pi_monte_carlo.m                  *)
(***********************************************************)
n = 10^7;
R = 1;
count=0;

t1 = AbsoluteTime[];
Do[ 

   X = Random[];
   Y = Random[];

   If[ (X^2 + Y^2) <= R^2,  
       count = count + 1;
       ];
   
   , {i,n} ];
t2 = AbsoluteTime[];
tt = t2 - t1;              (* Time in Do-loop *)

myPI = ( 4.0 * count ) / n;
exactPI = N [ Pi ];

Print[ exactPI ];
Print[ myPI ];
Print[ tt ];
Quit[ ]
