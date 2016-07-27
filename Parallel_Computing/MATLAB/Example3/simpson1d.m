%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Function: simpson1d.m
%
% Purpose: 1D integration - Simpson's 1/3 rule
%
%      f - function to integrate
%      a - lower bound   
%      b - upper bound
%
%      Must have odd number of data points
%
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function integral = simpson1d(f,a,b)

  num = length(f); % number of data points

  sc = 2*ones(num,1);
  sc(2:2:num-1) = 4;
  sc(1) = 1; sc(num) = 1;

  h = (b-a)/(num-1);

  integral = (h/3) * f * sc;

end