# VMEC++ Validation

## About the project

This project serves to validate [VMEC++](https://github.com/proximafusion/vmecpp), a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC), a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks against a reference VMEC implementation.

### Reference implementation
The reference implementation we compare against is Serial VMEC 8.52 from tag `v251` of [https://github.com/PrincetonUniversity/STELLOPT](https://github.com/PrincetonUniversity/STELLOPT) (sub-directory VMEC2000) with the following patches specified below for direct comparison with standalone VMEC++:
* `lnyquist` must be set to `.TRUE.` in `wrout.f`
* Parameter ordering when calling `analysum2` and `analysum2_par` must be fixed

A Docker image that contains serial VMEC 8.52 with the patches specified above is available as a Docker image at [ghcr.io/proximafusion/vmec2000:latest](ghcr.io/proximafusion/vmec2000:latest) in order to also freeze all dependent libraries, compiler versions, etc.

### Input configurations

#### Configurations

#### Parameter scans

### Error tolerance

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