%=================================================
% Program: prun.m
% Purpose: Parallel driver for se_fdm.m
%=================================================
% create a local cluster object
pc = parcluster('local')
N = 4;
parpool(pc, N)
parameter_list = [-295, -200, -150, -100]; % Energies: Ev
parfor i = 1: N
  se_fdm(parameter_list(i),i);
end
delete(gcp);
exit;
