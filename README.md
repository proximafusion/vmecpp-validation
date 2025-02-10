# VMEC++ Validation

[![Run short version of VMEC++ validation](https://github.com/proximafusion/vmecpp-validation/actions/workflows/short_validation.yaml/badge.svg?branch=main)](https://github.com/proximafusion/vmecpp-validation/actions/workflows/short_validation.yaml)

This project serves to validate [VMEC++](https://github.com/proximafusion/vmecpp), a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC), a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks against a reference VMEC implementation.

We compare the contents of the "wout" file, VMEC's standard output format, ensuring the values produced by VMEC++ match those of the reference implementation within test tolerances. 

Please report any issues at https://github.com/proximafusion/vmecpp.

## Usage

Instructions are for Linux systems, tested on Ubuntu 22.04 with Python 3.10.

### Install pre-requisites

```shell
sudo apt install python3-venv git-lfs

# VMEC++ pre-requisites:
sudo apt-get install build-essential cmake libnetcdf-dev liblapacke-dev libopenmpi-dev libeigen3-dev nlohmann-json3-dev libhdf5-dev
```

### Set up virtual environment

```shell
git clone https://github.com/proximafusion/vmecpp-validation.git
cd vmecpp-validation
python -m venv venv
source venv/bin/activate  # or equivalent for other shells than bash
pip install -r requirements.txt
```

### Run the validation

```shell
# activate virtual environment
source venv/bin/activate

env OMP_NUM_THREADS=1 python validate_vmec.py
```

`OMP_NUM_THREADS=1` guarantees that VMEC++ uses only one core like Fortran VMEC does.

Results will be saved in the current working durectory, in a subdirectory with the prefix `vnvresults`.

## About the project

### Reference VMEC implementation
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
* Matt Landreman’s preciseQA and QH configurations, to probe other classes of stellarator symmetries.
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
Unless explicitly specified in this document, our implementation acts as the specification in regards to what exact tolerances are used for each quantity: see `src/tolerances.py`.

As a rule of thumb, deviations from the Reference should be smaller than 1e-6 in the IsCloseRelAbs metric.
