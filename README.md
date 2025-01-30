# VMEC++ Validation

## About the project

This project serves to validate [VMEC++](https://github.com/proximafusion/vmecpp), a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC), a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks against a reference VMEC implementation.

### Reference implementation
The reference implementation we compare against is Serial VMEC 8.52 from tag `v251` of [https://github.com/PrincetonUniversity/STELLOPT](https://github.com/PrincetonUniversity/STELLOPT) (sub-directory VMEC2000) with the following patches specified below for direct comparison with standalone VMEC++:
* `lnyquist` must be set to `.TRUE.` in `wrout.f`
* Parameter ordering when calling `analysum2` and `analysum2_par` must be fixed

A Docker image that contains serial VMEC 8.52 with the patches specified above is available as a Docker image at [https://ghcr.io/proximafusion/vmec2000:latest](https://ghcr.io/proximafusion/vmec2000:latest) in order to also freeze all dependent libraries, compiler versions, etc.

### Input configurations

We run on 9 different plasma configurations with a 3-step parameter scan in beta, Fourier resolution and radial resolution (27 parameter combinations per configuration), excluding 24 that are known to not converge (see `src.input_generation.EXCLUDED_CONFIGURATIONS`); resulting in 219 total input configurations.

#### Configurations

We use the following input configurations:
* DSHAPE, as in the original VMEC paper, [Hirshman SP, Whitson JC. Steepest descent moment method for three-dimensional magnetohydrodynamic equilibria. (1983)](https://doi.org/10.1063/1.864116)
* HELIOTRON, as in the original VMEC paper, [Hirshman SP, Whitson JC. Steepest descent moment method for three-dimensional magnetohydrodynamic equilibria. (1983)](https://doi.org/10.1063/1.864116)
* Configurations probing the region of interest for Proxima:
    * CM-A
    * CM-B
* Matt’s preciseQA and QH configurations, to probe other classes of stellarator symmetries.
    * preciseQA (take the “20211102-01-precise_quasisymmetry_zenodo/configurations/new_QA_well/input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered” configuration in this [Zenodo folder](https://zenodo.org/records/5645413))
    * preciseQH (take the “20211102-01-precise_quasisymmetry_zenodo/configurations/new_QH_well/input.20210728-01-026_QH_nfp4_A8_magwell” configuration in this [Zenodo folder](https://zenodo.org/records/5645413))
* Some existing machines, as it is likely that we can obtain or have equilibria computed by other codes to be possibly used for benchmarking.
    * CTH-like (test.vmec from Stellarator-Tools)
    * W7-X standard (EIM)
    * NCSX (from STELLOPT wiki)

For existing machines (CTH-like, W7-X and NCSX), we compare free-boundary runs. For other configurations, we use a fixed-boundary run.

#### Parameter scans

We use the following parameters for each configuration:
* Beta
    * 0% -> vacuum
    * 1% -> low-pressure
    * 5% -> high-pressure
* Fourier resolution (i.e., mpol and ntor); keep mpol = ntor for stellarator cases
    * low-res: 5
    * medium-res: 7
    * high-res: 12
* Radial resolution (i.e., ns)
    * low-res: 20
    * medium-res: 51
    * high-res: 99

### Validation checks

For each input configuration, the following checks must pass:
* The implementation converges
    * Normal termination
    * Number of iteration similar to the reference
* For fixed-boundary runs, single-thread runtime is not more than 10% higher than the Reference or not more than 1s longer (so slowdowns larger than 10% are allowed for sub-second runtimes, where errors are larger)
* Location of inner flux surfaces and magnetic field components in all (e.g. 201) inner flux surface of a given boundary are lower than 1e-4 absolute difference or lower for normalized field strength when evaluated against reference
* Quantities from the "wout" output file match. For a full list of quantities, see `src/tolerances.py`


### Error tolerance

Unless specified otherwise, quantities are checked against the IsCloseRelAbs metric (see Gill, Murray and Wright, "Practical Optimization" (1984), sec. 2.1.1, lower formula on p. 7).
Unless explicitly specified in this document, our implementation of the V&V checks acts as the specification in regards to what exact tolerances are used for each quantity: see `src/tolerances.py`.

As a rule of thumb, deviations from the Reference should be smaller than 1e-6 in the IsCloseRelAbs metric.

## Getting started

Installation and usage instructions are for Linux systems, tested on Ubuntu 22.04.

### Pre-requisites:
* python with venv, tested with python3.10
```
sudo apt install python3-venv
```
* system packages required by vmecpp, e.g.
```
sudo apt-get install build-essential cmake libnetcdf-dev liblapacke-dev libopenmpi-dev libeigen3-dev nlohmann-json3-dev libhdf5-dev
```
* [git-lfs](https://git-lfs.com/), when working with free-boundary scenarios (currently w7x, ncsx and cth_like)

### Installation

1. Clone the repo
2. Set up python virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

And you are good to go!

## Usage

If needed, activate your virtual environment:
```
source venv/bin/activate
```

Run the validation:
```
python -m validate_vmec
```

Results will be in the repo root, in a directory with the prefix `vnvresults`.

TODO(viska) Add more background information and hints to interpret the results