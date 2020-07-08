extern crate ising_monte_carlo;
use self::ising_monte_carlo::parallel_tempering::autocorrelations::ParallelTemperingAutocorrelations;
use ising_monte_carlo::graph::*;
use ising_monte_carlo::parallel_tempering::*;
use ndarray::{Array, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
pub struct LatticeTempering {
    tempering: DefaultTemperingContainer<ThreadRng, SmallRng>,
}

#[pymethods]
impl LatticeTempering {
    #[new]
    fn new(edges: Vec<(Edge, f64)>) -> Self {
        let nvars = edges
            .iter()
            .map(|((a, b), _)| max(*a, *b))
            .max()
            .map(|x| x + 1)
            .unwrap();
        let cutoff = nvars;

        let rng = rand::thread_rng();
        let tempering =
            DefaultTemperingContainer::<ThreadRng, SmallRng>::new(rng, edges, cutoff, false, false);
        Self { tempering }
    }

    fn add_graph(&mut self, transverse: f64, beta: f64) {
        let rng = SmallRng::from_entropy();
        self.tempering.add_graph(rng, transverse, beta);
    }

    fn qmc_timesteps(&mut self, t: usize) {
        self.tempering.parallel_timesteps(t)
    }

    fn qmc_timesteps_sample(
        &mut self,
        py: Python,
        timesteps: usize,
        replica_swap_freq: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> PyResult<(Py<PyArray3<bool>>, Py<PyArray1<f64>>)> {
        let sampling_freq = sampling_freq.unwrap_or(1);
        let replica_swap_freq = replica_swap_freq.unwrap_or(1);
        let mut states = Array3::<bool>::default((
            self.tempering.num_graphs(),
            timesteps / sampling_freq,
            self.tempering.nvars(),
        ));
        let mut energy_acc = vec![0.0; self.tempering.num_graphs()];

        let mut remaining_timesteps = timesteps;
        let mut time_to_swap = replica_swap_freq;
        let mut time_to_sample = sampling_freq;
        let mut timestep_index = 0;

        while remaining_timesteps > 0 {
            let t = min(min(time_to_sample, time_to_swap), remaining_timesteps);
            self.tempering
                .graph_mut()
                .par_iter_mut()
                .map(|(g, beta)| g.timesteps(t, *beta))
                .zip(energy_acc.par_iter_mut())
                .for_each(|(te, e)| {
                    *e += te * t as f64;
                });
            time_to_sample -= t;
            time_to_swap -= t;
            remaining_timesteps -= t;

            if time_to_swap == 0 {
                self.tempering.parallel_tempering_step();
                time_to_swap = replica_swap_freq;
            }
            if time_to_sample == 0 {
                let graphs = self.tempering.graph_ref().par_iter().map(|(g, _)| g);
                states
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip(graphs)
                    .for_each(|(mut s, g)| {
                        s.index_axis_mut(ndarray::Axis(0), timestep_index)
                            .iter_mut()
                            .zip(g.state_ref().iter())
                            .for_each(|(a, b)| {
                                *a = *b;
                            })
                    });
                timestep_index += 1;
                time_to_sample = sampling_freq;
            }
        }
        let py_states = states.into_pyarray(py).to_owned();

        let energies = energy_acc
            .into_iter()
            .map(|e| e / timesteps as f64)
            .collect::<Vec<_>>();
        let energies = Array::from(energies);
        let py_energies = energies.into_pyarray(py).to_owned();
        Ok((py_states, py_energies))
    }

    /// Run a quantum monte carlo simulation, get the variable's autocorrelation across time for
    /// each experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_variable_autocorrelation(
        &mut self,
        py: Python,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        replica_swap_freq: Option<usize>,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        if sampling_wait_buffer > 0 {
            self.tempering.parallel_timesteps(sampling_wait_buffer);
        }

        let autos = self.tempering.calculate_variable_autocorrelation(
            timesteps,
            replica_swap_freq,
            sampling_freq,
            use_fft,
        );
        let mut corrs = Array::default((self.tempering.num_graphs(), timesteps));
        corrs
            .axis_iter_mut(ndarray::Axis(0))
            .zip(autos.into_iter())
            .for_each(|(mut s, auto)| s.iter_mut().zip(auto.into_iter()).for_each(|(a, b)| *a = b));

        let py_corrs = corrs.into_pyarray(py).to_owned();
        Ok(py_corrs)
    }

    /// Run a quantum monte carlo simulation, get the bond's autocorrelation across time for each
    /// experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_bond_autocorrelation(
        &mut self,
        py: Python,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        replica_swap_freq: Option<usize>,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        if sampling_wait_buffer > 0 {
            self.tempering.parallel_timesteps(sampling_wait_buffer);
        }

        let autos = self.tempering.calculate_bond_autocorrelation(
            timesteps,
            replica_swap_freq,
            sampling_freq,
            use_fft,
        );
        let mut corrs = Array::default((self.tempering.num_graphs(), timesteps));
        corrs
            .axis_iter_mut(ndarray::Axis(0))
            .zip(autos.into_iter())
            .for_each(|(mut s, auto)| s.iter_mut().zip(auto.into_iter()).for_each(|(a, b)| *a = b));

        let py_corrs = corrs.into_pyarray(py).to_owned();
        Ok(py_corrs)
    }
}
