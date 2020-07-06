extern crate ising_monte_carlo;
use self::ising_monte_carlo::sse::fast_ops::{FastOpNode, FastOps};
use ising_monte_carlo::graph::*;
use ising_monte_carlo::parallel_tempering::*;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use rand::prelude::*;
use std::cmp::{min, max};

#[pyclass]
pub struct LatticeTempering {
    tempering: TemperingContainer<ThreadRng, SmallRng, FastOpNode, FastOps, FastOps>,
}

#[pymethods]
impl LatticeTempering {
    #[new]
    fn new(
        edges: Vec<(Edge, f64)>,
    ) -> Self {
        let nvars = edges
            .iter()
            .map(|((a, b), _)| max(*a, *b))
            .max()
            .map(|x| x + 1)
            .unwrap();
        let cutoff = nvars;

        let rng = rand::thread_rng();
        let tempering =
            TemperingContainer::<ThreadRng, SmallRng, FastOpNode, FastOps, FastOps>::new(
                rng,
                edges,
                cutoff,
                false,
                false,
            );
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
        mut timesteps: usize,
        replica_swap_freq: usize,
        sampling_freq: Option<usize>,
    ) -> PyResult<Py<PyArray3<bool>>> {
        let sampling_freq = sampling_freq.unwrap_or(1);
        let mut states = Array3::<bool>::default((
            self.tempering.num_graphs(),
            timesteps / sampling_freq,
            self.tempering.nvars(),
        ));

        let mut time_to_swap = replica_swap_freq;
        let mut time_to_sample = sampling_freq;
        let mut timestep_index = 0;

        while timesteps > 0 {
            let t = min(time_to_sample, time_to_swap);
            self.tempering.parallel_timesteps(t);
            time_to_sample -= t;
            time_to_swap -= t;
            timesteps -= t;

            if time_to_swap == 0 {
                self.tempering.parallel_tempering_step();
                time_to_swap = replica_swap_freq;
            }
            if time_to_sample == 0 {
                states
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_iter()
                    .zip(self.tempering.graph_ref().into_iter())
                    .for_each(|(mut s, (g, _))| {
                        let mut s = s.index_axis_mut(ndarray::Axis(0), timestep_index);
                        let state = g.state_ref();
                        s.iter_mut().zip(state.iter()).for_each(|(a, b)| {
                            *a = *b;
                        })

                    });
                timestep_index += 1;
                time_to_sample = sampling_freq;
            }
        }
        let py_states = states.into_pyarray(py).to_owned();
        Ok(py_states)
    }
}
