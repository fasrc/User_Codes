%===============================================================
% Program: rnd_test(n, a ,b, iseed)
%          Create a random vector and sum up its elements
%===============================================================
function [] = rnd_test(n, a, b, iseed)
  % Create random vector vec
  vec = rand_vec(n, a, b, iseed);

  % Sum up elements of vec
  s = vec_sum(n, vec);

  % Print out results
  r = sprintf('%f ', vec);
  fprintf('Random vector: [ %s ]\n', r);
  fprintf('Sum of elements: %f\n', s);
end

%===============================================================
% Function: rand_vec(n, a ,b)
%           Create a random vector x of dimension n with
%           uniformly distributed numbers in the interval [a, b]
%===============================================================
function x = rand_vec(n, a, b, iseed)
  rng(iseed);
  x = a + ( b - a ) * rand(n,1);
end

%===============================================================
% Function: vec_sum(n, vec)
%           Sum up elements of vector vec with dimension n
%===============================================================
function s = vec_sum(n, vec)
  s = 0.0;
  for i = 1:n
    s = s + vec(i,1);
  end
end
