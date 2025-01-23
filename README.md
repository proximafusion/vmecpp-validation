# VMEC++ Validation

## Pre-requisites:
* python with venv, tested with python3.10
```
sudo apt install python3-venv
```
* system packages required by vmecpp, e.g.
```
sudo apt-get install build-essential cmake libnetcdf-dev liblapacke-dev libopenmpi-dev libeigen3-dev nlohmann-json3-dev libhdf5-dev
```

## Installation

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
