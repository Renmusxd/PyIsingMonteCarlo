# PyIsingMonteCarlo

This is the repo for the python bindings to the [`ising_monte_carlo`](https://github.com/Renmusxd/IsingMonteCarlo) 
rust package.

## Installation
### Install Rust:
The backend for simulations is written in rust, a compiled language with strong type and memory security guaranties. To
use this library you must first install the rust compiler by following the instructions here:

https://www.rust-lang.org/tools/install

And specifically the `nightly` toolchain to get access to some of the more advanced language features:

`$ rustup toolchain install nightly`

### Setup your python3 virtualenv
Setting up a virtualenv will depend on your system, but if you're unsure you should be able to run 
`pip install virtualenv` to get the virtualenv tool. Then make the virtualenv where you'd like, and install some 
required packages: 

```
$ virtualenv --python=python3 <path>
$ source <path>/bin/activate
$ pip install numpy maturin
```

### Build and install the package
Navigate to this cloned repository, and build the _release_ version of the library (meaning highly optimized).

```
$ git clone https://github.com/Renmusxd/PyIsingMonteCarlo
$ cd PyIsingMonteCarlo
$ maturin develop --release --strip
```

## Building a wheel
If you'd prefer to make a wheel to install later, you can use the supplied makefile which runs

`$ maturin build --release --strip --no-sdist`

## Usage

To use the library, import `py_monte_carlo` at the top of your python file, then build a lattice to simulate. 
The lattice takes a list of edges, each is _((a, b), j)_ where _a_ and _b_ are variable indices, and _j_ is the bond. 
It's of the form `J*Sza*Szb` so positive is antiferromagnetic.

```python
import py_monte_carlo
edges = [
    ((0, 1), 1.0),
    ((1, 2), -1.0)
]
lat = py_monte_carlo.Lattice(edges)
```

You may then run classical monte carlo using any of the method:
```python
lat.run_monte_carlo(beta, timesteps, num_experiments)
lat.run_monte_carlo_sampling(beta, timesteps, num_experiments)
lat.run_monte_carlo_annealing(betas, timesteps, num_experiments)
lat.run_monte_carlo_annealing_and_get_energies(betas, timesteps, num_experiments)
```

Alternatively you can add a transverse field: `lat.set_transverse_field(gamma)` and run quantum monte carlo:

```python
lat.run_quantum_monte_carlo(beta, timesteps, num_experiments)
lat.run_quantum_monte_carlo_sampling(beta, timesteps, num_experiments)
lat.run_quantum_monte_carlo_and_measure_spins(beta, timesteps, num_experiments)
```

as well as get correlation functions over time for either individual variables or broken bonds:

```python
lat.run_quantum_monte_carlo_and_measure_variable_autocorrelation(beta, timesteps, num_experiments)
lat.run_quantum_monte_carlo_and_measure_bond_autocorrelation(beta, timesteps, num_experiments)
```

All of these functions have additional optional arguments, see docs for additional details.