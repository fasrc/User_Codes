%=====================================================================
% Program: knitro_test.m
%          Program illustrates use of Knitro with MATLAB
%=====================================================================

% objective to minimize
obj = @(x) 1000 - x(1)^2 - 2*x(2)^2 - x(3)^2 - x(1)*x(2) - x(1)*x(3);

% No nonlinear equality constraints.
ceq  = [];

% Specify nonlinear inequality constraint to be nonnegative
c2 =  @(x) x(1)^2 + x(2)^2 + x(3)^2 - 25;

% "nlcon" should return [c, ceq] with c(x) <= 0 and ceq(x) = 0
% so we need to negate the inequality constraint above
nlcon = @(x)deal(-c2(x), ceq);

% Initial point
x0  = [2; 2; 2];

% No linear inequality contraint ("A*x <= b")
A = [];
b = [];

% Since the equality constraint "c1" is linear, specify it here  ("Aeq*x = beq")
Aeq = [8 14 7];
beq = [56];

% lower and upper bounds
lb = zeros(3,1);
ub = [];

% solver call
x = knitro_nlp(obj, x0, A, b, Aeq, beq, lb, ub, nlcon);
