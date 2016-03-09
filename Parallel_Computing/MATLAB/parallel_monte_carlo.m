%===========================================================================
% Parallel Monte Carlo calculation of PI
%===========================================================================
function [] = parallel_monte_carlo(nproc)
  parpool('local', nproc)
  R = 1;
  darts = 1e7;
  count = 0;
  tic
  parfor i = 1:darts
    % Compute the X and Y coordinates of where the dart hit the...............
    % square using Uniform distribution.......................................
    x = R*rand(1);
    y = R*rand(1);
    if x^2 + y^2 <= R^2
      % Increment the count of darts that fell inside of the.................
      % circle...............................................................
      count = count + 1; % Count is a reduction variable.
    end
  end
  % Compute pi.................................................................
  myPI = 4*count/darts;
  T = toc;
  fprintf('The computed value of pi is %8.7f.\n',myPI);
  fprintf('The parallel Monte-Carlo method is executed in %8.2f seconds.\n', T);
  delete(gcp);
end
