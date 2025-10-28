# Notes Directory - Informal Advanced HPC Software Installation & Configuration Guides

One day the contents herein may be promoted to actual entries.

## Overview

This directory contains comprehensive documentation for installing, configuring, and troubleshooting specialized software packages on Harvard's Research Computing (RC) cluster environment. These notes represent field-tested procedures, workarounds, and best practices that supplement the official documentation, particularly for complex installations requiring specific dependencies, version compatibility, or non-standard configurations.

## Directory Purpose

The **Notes** directory serves as a knowledge repository for:
- 🔧 Complex software installation procedures not covered in standard documentation
- 🐛 Troubleshooting guides and known workarounds for common issues
- 📊 Performance optimization tips and resource allocation strategies
- 🔬 Specialized scientific computing workflows and integrations
- 💡 Best practices for HPC software deployment and management

## Content Organization

### 📁 Directory Structure

```
Notes/
├── Machine Learning & AI Tools/
│   ├── alphapose.md           # Multi-person pose estimation
│   ├── mmpose.md              # OpenMMLab pose estimation framework
│   ├── JAX.md                 # Accelerated array computing
│   ├── ollama.md              # Local LLM deployment
│   └── gReLU/grelu.md         # Genomics ML tool
├── Scientific Computing/
│   ├── amber.md               # Molecular dynamics
│   ├── qchem.md               # Quantum chemistry
│   ├── OpenFoam/              # CFD simulations
│   └── cutlass.md             # CUDA linear algebra
├── Statistical Computing/
│   ├── INLA_R.md              # Bayesian inference for R
│   └── R_in_Jupyter.md        # R kernel setup
├── Development Tools/
│   ├── vscode_remote_tunnel.md # Remote development
│   └── augustus-install.md    # Bioinformatics tool
├── Workflow Management/
│   └── JobArrays/             # SLURM array job examples
└── Reference Documents/
    ├── *.pdf                  # GUI guides and tutorials
    └── plotly_test.ipynb      # Visualization examples
```

## Detailed Documentation Index

### 🤖 Machine Learning & AI Tools

#### **AlphaPose** ([alphapose.md](alphapose.md))
Multi-person pose estimation framework with real-time capabilities.
- **Installation Method**: Conda environment with GPU support
- **Key Requirements**: CUDA 11.3.1, Python 3.7, GPU partition
- **Steps**: 7-step process including dependency compilation
- **Use Cases**: Human pose detection, motion tracking, biomechanics research
- **Special Notes**: Requires manual compilation of some dependencies

#### **MMPose** ([mmpose.md](mmpose.md))
Part of OpenMMLab ecosystem for pose estimation research.
- **Installation Method**: Source compilation with mim package manager
- **Key Requirements**: PyTorch 2.3.0, CUDA 12.1, 32GB+ RAM
- **Steps**: 9-step installation with mmcv compilation
- **Use Cases**: Research-grade pose estimation, benchmark evaluations
- **Testing**: Includes GPU validation procedures

#### **JAX** ([JAX.md](JAX.md))
Google's accelerated NumPy for machine learning research.
- **Installation Methods**: 
  - Conda with system modules
  - Self-contained conda environment
  - Singularity containers
- **Key Requirements**: CUDA 12.4/12.5, cuDNN 9.1.0
- **Performance**: Includes benchmarking examples
- **Use Cases**: Neural network research, scientific computing, automatic differentiation

#### **Ollama** ([ollama.md](ollama.md))
Run large language models locally on HPC infrastructure.
- **Installation Method**: Singularity container deployment
- **Architecture**: Client-server model with two-terminal setup
- **Models Supported**: Llama3, Mistral, others
- **Use Cases**: Private LLM inference, model evaluation, research applications
- **Resource Requirements**: GPU recommended, 16GB+ RAM minimum

#### **gReLU** ([gReLU/grelu.md](gReLU/grelu.md))
Genomics-specific machine learning tool for regulatory element analysis.
- **Installation Method**: Conda environment with Jupyter integration
- **Steps**: 11-step process including kernel configuration
- **Use Cases**: Genomic sequence analysis, regulatory element prediction
- **Integration**: Works with Jupyter Lab for interactive analysis

### 🧬 Scientific Computing & Simulation

#### **Amber22** ([amber.md](amber.md))
Molecular dynamics simulation package for biomolecules.
- **Build Variants**: 
  - Serial version (single-core)
  - MPI+CUDA version (multi-GPU)
- **Compiler Requirements**: Intel 2021.2, CUDA 11.7.1
- **Build Time**: ~2 hours for full compilation
- **Use Cases**: Protein dynamics, drug discovery, materials science
- **Known Issues**: Version compatibility warnings documented

#### **Q-Chem** ([qchem.md](qchem.md))
Quantum chemistry software for electronic structure calculations.
- **Installation Method**: Module-based (pre-installed)
- **Module**: QChem/6.1-fasrc01
- **Features**: DFT, post-HF methods, excited states
- **Job Submission**: SLURM examples provided
- **Input Examples**: Sample calculation files included

#### **OpenFOAM with RheoTool** ([OpenFoam/README.md](OpenFoam/README.md))
Computational fluid dynamics for complex rheological flows.
- **Installation Method**: Singularity container
- **Dependencies**: PETSc, Eigen, specialized solvers
- **Use Cases**: Non-Newtonian fluids, viscoelastic flows
- **Special Features**: RheoTool extension for rheology
- **Container Management**: Complete workflow documented

#### **NVIDIA CUTLASS** ([cutlass.md](cutlass.md))
CUDA templates for linear algebra subroutines.
- **Installation Method**: Source compilation
- **Build Requirements**: 100GB+ disk space, 4+ hours compilation
- **GPU Support**: Optimized for V100, A100
- **Use Cases**: Deep learning primitives, HPC kernels
- **Testing**: Extensive test suite (may require memory management)

### 📊 Statistical Computing & Data Analysis

#### **R-INLA** ([INLA_R.md](INLA_R.md))
Integrated Nested Laplace Approximation for Bayesian inference.
- **Critical**: Version compatibility matrix provided
- **R Versions Tested**: 4.0.5, 4.2.2.1, 4.3.1, 4.4.0
- **Installation**: Specific procedures to avoid crashes
- **Use Cases**: Spatial statistics, hierarchical models
- **Troubleshooting**: Known issues and workarounds documented

#### **R in Jupyter** ([R_in_Jupyter.md](R_in_Jupyter.md))
Configure R kernel for Jupyter notebook environments.
- **Setup Method**: IRkernel installation
- **Integration**: OpenOnDemand VDI portal
- **Path Configuration**: Custom library paths
- **Use Cases**: Interactive data analysis, reproducible research
- **Scripts**: Setup automation provided

### 🛠️ Development Tools & Infrastructure

#### **VS Code Remote Tunnel** ([vscode_remote_tunnel.md](vscode_remote_tunnel.md))
Enable VS Code development on compute nodes.
- **Connection Method**: CLI-based tunnel
- **Authentication**: GitHub/Microsoft account required
- **Features**: Full IDE experience on HPC
- **Workflow**: Direct compute node access
- **Benefits**: GPU debugging, large memory operations

#### **Augustus** ([augustus-install.md](augustus-install.md))
Gene prediction tool for eukaryotic genomes.
- **Installation Method**: Spack package manager
- **Steps**: 3-command installation
- **Testing**: Validation procedures included
- **Use Cases**: Genome annotation, comparative genomics
- **Dependencies**: Handled automatically by Spack

### 📋 Workflow Management

#### **SLURM Job Arrays** ([JobArrays/](JobArrays/))
Comprehensive examples for parallel job submission patterns.

- **Exercise Types**:
  - `Exercise_sequential_basic/`: Simple indexed arrays
  - `Exercise_recursive/`: Nested job dependencies
  - `Exercise_nonsequential/`: Custom index patterns
  - `Exercise_nonsequential_maxarray/`: Large-scale arrays

- **Key Features**:
  - FastQC bioinformatics workflow integration
  - Python script coordination
  - Resource optimization strategies
  - Error handling patterns

- **Scripts Included**:
  - `recursive-array.sh`: Automated array job generation
  - Sample data structures for testing
  - Command reference guide

### 📚 Reference Documents (PDFs)

#### **Dash-Cannon.pdf**
Complete guide for deploying Dash/Plotly web applications on the cluster.
- Interactive dashboard creation
- Port forwarding setup
- Security considerations

#### **Flask-Cannon.pdf**
Flask web application deployment on HPC infrastructure.
- Development to production workflow
- Resource allocation strategies
- Scaling considerations

#### **Stata-Python-Instructions.pdf**
Integration guide for Stata with Python environments.
- Jupyter kernel configuration
- Data exchange patterns
- Mixed-language workflows

#### **Tidyverse-R-CLI.pdf**
Command-line R with tidyverse packages.
- Non-interactive execution
- Batch processing strategies
- Performance optimization

#### **Devtools-R-terminal.pdf**
R package development in terminal environments.
- Build and test workflows
- Dependency management
- CI/CD integration

### 📊 Interactive Examples

#### **plotly_test.ipynb**
Jupyter notebook demonstrating Plotly visualizations on HPC.
- 3D plotting examples
- Large dataset handling
- Export strategies for publications
- Performance considerations for remote rendering

## Installation Methods Overview

### 🔍 Method Comparison Table

| Method | Best For | Pros | Cons | Examples |
|--------|----------|------|------|----------|
| **Modules** | Stable, widely-used software | Pre-configured, optimized, maintained | Limited versions, less flexibility | Q-Chem, standard R |
| **Conda/Mamba** | Python ecosystems, ML frameworks | Environment isolation, dependency resolution | Storage overhead, activation time | JAX, AlphaPose |
| **Source Compilation** | Performance-critical, custom features | Maximum optimization, latest versions | Time-consuming, complex dependencies | Amber, CUTLASS |
| **Singularity** | Complex dependencies, reproducibility | Portable, consistent, shareable | Container overhead, limited host access | Ollama, OpenFOAM |
| **Spack** | Scientific software, HPC tools | Automatic optimization, dependency handling | Learning curve, build time | Augustus |

## Common Patterns & Best Practices

### 🎯 Resource Allocation Guidelines

#### GPU Jobs
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
```

#### Memory-Intensive Tasks
```bash
#SBATCH --partition=bigmem
#SBATCH --mem=256GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
```

### 🔧 Environment Management

#### Module Loading Order
1. Compiler (gcc/intel)
2. MPI implementation (if needed)
3. CUDA (for GPU applications)
4. Application-specific modules

#### Conda Best Practices
```bash
# Use mamba for faster dependency resolution
module load python
mamba create -n myenv python=3.10
source activate myenv

# Clean cache regularly
conda clean --all
```

### 🐛 Common Troubleshooting

#### CUDA Version Mismatches
- Check `nvidia-smi` vs module versions
- Use `module spider cuda` to find compatible versions
- Consider container solutions for complex dependencies

#### Memory Errors
- Increase SLURM memory allocation
- Check `ulimit -a` for system limits
- Monitor with `sstat -j $SLURM_JOB_ID`

#### Library Path Issues
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libs
export PYTHONPATH=$PYTHONPATH:/path/to/packages
```

## Quick Reference Commands

### Module Management
```bash
module avail [software]     # List available versions
module spider [software]    # Search all modules
module load [module]        # Load module
module list                 # Show loaded modules
module purge               # Unload all modules
```

### SLURM Commands
```bash
sinfo -p gpu               # Check GPU availability
squeue -u $USER           # View your jobs
scancel [jobid]           # Cancel job
sstat -j [jobid]          # Job statistics
sacct -j [jobid]          # Job accounting info
```

### Container Commands
```bash
singularity pull docker://[image]  # Download container
singularity shell [image]          # Interactive shell
singularity exec [image] [command] # Execute command
```

## Contributing Guidelines

When adding new documentation to this directory:

1. **File Naming**: Use lowercase with underscores (e.g., `new_software.md`)
2. **Structure**: Include:
   - Brief description
   - Requirements/prerequisites
   - Step-by-step installation
   - Testing/validation
   - Common issues
   - Example usage

3. **Version Information**: Always specify:
   - Software version
   - Module versions used
   - Date of last test
   - Cluster partition tested on

4. **Code Blocks**: Use language-specific syntax highlighting
5. **Error Messages**: Include full error text for searchability

## Maintenance Notes

- **Last Review**: Documentation current as of repository snapshot
- **Update Frequency**: As new software is tested and deployed
- **Validation**: All procedures tested on FASRC Cannon cluster
- **Compatibility**: Procedures may need adjustment for other HPC systems

## Related Resources

- [FASRC User Documentation](https://docs.rc.fas.harvard.edu/)
- [Main User_Codes Repository](../)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [Singularity User Guide](https://sylabs.io/guides/latest/user-guide/)
- [Spack Package Manager](https://spack.io/)

## Support

For questions about these installation guides:
1. Check the specific document's troubleshooting section
2. Review FASRC documentation for general HPC issues
3. Submit a ticket to RC support for cluster-specific problems
4. Contribute improvements via pull requests

---

*This README provides an index to specialized software installation and configuration guides. Each document represents practical experience and solutions to real deployment challenges on HPC infrastructure.*