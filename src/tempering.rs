extern crate ising_monte_carlo;
use ising_monte_carlo::graph::*;
use ising_monte_carlo::parallel_tempering::*;
use ndarray::{Array, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3};
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
            self.tempering.graph_mut().par_iter_mut()
                .zip(energy_acc.par_iter_mut())
                .for_each(|((graph, beta), e)| {
                    *e += graph.timesteps(t, *beta) * t as f64;
                });
            time_to_sample -= t;
            time_to_swap -= t;
            remaining_timesteps -= t;

            if time_to_swap == 0 {
                self.tempering.parallel_tempering_step();
                time_to_swap = replica_swap_freq;
            }
            if time_to_sample == 0 {
                let graphs = self.tempering.graph_ref().par_iter();
                states
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip(graphs)
                    .for_each(|(mut s, (g, _))| {
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

        let energies = energy_acc.into_iter().map(|e| e / timesteps as f64).collect::<Vec<_>>();
        let energies = Array::from(energies);
        let py_energies = energies.into_pyarray(py).to_owned();
        Ok((py_states, py_energies))
    }
}
