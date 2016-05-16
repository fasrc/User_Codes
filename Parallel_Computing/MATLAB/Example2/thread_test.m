%===========================================================
% Program: thread_test.m
%          Multithreading test
%===========================================================
warning off;                % Turn off warnings

n = 4000;                   % set matrix size
A = rand(n);                % create random matrix
B = rand(n);                % create another random matrix

% Vector implementation of matrix-matrix multiplication
tic
for i=1:4
  nproc(i) = 2^(i-1);
  fprintf('Number of threads: %d\n', nproc(i));
  maxNumCompThreads(nproc(i)); % set the thread count to 1, 2, 4, or 8
  tic                          % starts timer
  C = A * B;                   % matrix multiplication
  walltime(i) = toc;           % wall clock time
  speedup(i) = walltime(1) / walltime(i);
  efficiency(i) = 100 * speedup(i) / (2^(i-1));
end

fprintf('\n');
fprintf('%10s  %s  %s  %s\n','Nproc','Walltime','Speedup','Efficiency (%)');
for i=1:4
  fprintf('%8d  %8.2f  %8.2f  %10.2f\n', nproc(i), walltime(i), speedup(i), efficiency(i));
end
