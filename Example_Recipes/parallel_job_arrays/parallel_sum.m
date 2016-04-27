%=====================================================================
% Function: parallel_sum( N )
%           Calculates integer sum from 1 to N in parallel
%=====================================================================
function [] = parallel_sum(infile)
  % Read input data
  fid = fopen(infile,'r');
  m = fgets(fid);
  fclose(fid);
  N = sscanf(m,'%d');

  % Create a local cluster object
  pc = parcluster('local');

  % explicitly set the JobStorageLocation to the temp directory that was
  % created in your sbatch script
  pc.JobStorageLocation = strcat('/scratch/', getenv('USER'),'/', getenv('SLURM_ARRAY_TASK_ID'))

  % Start local parallel pool
  parpool(pc, str2num(getenv('SLURM_NTASKS')))

  % Calculate sum in parallel with PARFOR
  s = 0;
  parfor i = 1:N
    s = s + i;
  end

  % Print out result
  fprintf('Sum of numbers from 1 to %d is %d.\n', N, s);

  % Shut down locall parallel pool
  delete(gcp)

end
