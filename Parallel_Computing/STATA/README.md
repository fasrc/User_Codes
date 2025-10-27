# STATA Parallel Computing Resources

Users can leverage these resources to implement parallel processing techniques in STATA for enhanced data analysis.

For general information about parallel computing on Harvard's cluster, see the [RC User Documentation](https://docs.rc.fas.harvard.edu/).

## Parallel Processing Techniques

### Batch Processing
Efficiently process large datasets by splitting tasks into manageable batches.

- **[Example1](Example1/)**: Demonstrates batch processing of data with parallel execution. Includes scripts for job submission and output analysis.
  - **[script.do](./Example1/script.do)**: Main STATA script for batch processing.
  - **[output.log](./Example1/output.log)**: Log file capturing execution details.
  - **[results.dta](./Example1/results.dta)**: Sample output dataset for analysis.

### Simulation Studies  
Run multiple simulations to assess variability and robustness in findings.

- **[SimulationExample](SimulationExample/)**: Illustrates conducting parallel simulations in STATA to evaluate statistical models.
  - **[simulations.do](./SimulationExample/simulations.do)**: STATA script for running simulations in parallel.
  - **[summary_results.dta](./SimulationExample/summary_results.dta)**: Compiled results from simulation runs.

## Advanced Statistical Techniques

### Parallelized Estimation  
Speed up estimation processes for complex models using parallel computing.

- **[EstimationExample](EstimationExample/)**: Example of parallelized estimation techniques to improve computational efficiency.
  - **[estimation.do](./EstimationExample/estimation.do)**: STATA script for parallel estimation procedures.
  - **[estimation_results.dta](./EstimationExample/estimation_results.dta)**: Output dataset from the estimation processes.

## Quick Reference

| Technique             | Best For                       | Key Features                      | Example Files                     |
|----------------------|--------------------------------|-----------------------------------|-----------------------------------|
| **Batch Processing**  | Large datasets                 | Task splitting, job scheduling    | [script.do](./Example1/script.do) |
| **Simulation Studies**| Assessing variability          | Running multiple simulations       | [simulations.do](./SimulationExample/simulations.do) |
| **Parallelized Estimation** | Complex statistical models | Speeding up estimation processes   | [estimation.do](./EstimationExample/estimation.do) |