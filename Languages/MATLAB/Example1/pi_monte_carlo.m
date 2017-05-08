%=====================================================================
% Program: pi_monte_carlo.m
%
%          Parallel Monte Carlo calculation of PI
%
% Run:     matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"
%=====================================================================
R = 1.0;
darts = 1e6;
count = 0;
for i = 1:darts
  % Compute the X and Y coordinates of where the dart hit the.........
  % square using Uniform distribution.................................
  x = R*rand(1);
  y = R*rand(1);
  if x^2 + y^2 <= R^2
    % Increment the count of darts that fell inside of the circle.....
    count = count + 1; % Count is a reduction variable.
  end
end
% Compute pi..........................................................
myPI = 4*count/darts;
fprintf('The computed value of pi is %8.7f.\n',myPI);
exit;
