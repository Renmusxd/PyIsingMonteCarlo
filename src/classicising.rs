use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use qmc::classical::graph::GraphState;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::max;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
#[derive(Clone)]
pub struct ClassicIsing {
    nvars: usize,
    edges: Vec<((usize, usize), f64)>,
    longitudinal: f64,
    graphs: Vec<GraphState<SmallRng>>,
    rng: SmallRng,
    use_basic_moves: bool,
    seed: Option<u64>,
}

#[pymethods]
impl ClassicIsing {
    /// Construct a new instance.
    #[new]
    fn new(
        edges: Vec<((usize, usize), f64)>,
        longitudinal: Option<f64>,
        num_experiments: Option<usize>,
        seed: Option<u64>,
        use_basic_moves: Option<bool>,
    ) -> Self {
        let nvars = edges
            .iter()
            .map(|((a, b), _)| max(*a, *b))
            .max()
            .map(|x| x + 1)
            .unwrap();
        let rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let use_basic_moves = use_basic_moves.unwrap_or(false);
        let longitudinal = longitudinal.unwrap_or(0.0);
        let num_experiments = num_experiments.unwrap_or(1);
        let mut s = Self {
            nvars,
            edges,
            longitudinal,
            graphs: Default::default(),
            rng,
            use_basic_moves,
            seed,
        };
        (0..num_experiments).for_each(|_| s.add_graph(None, None));
        s
    }

    /// Add a new experiment.
    fn add_graph(
        &mut self,
        initial_state: Option<Vec<bool>>,
        edge_move_importance_sampling: Option<bool>,
    ) {
        let seed = self.rng.gen();
        let rng = SmallRng::seed_from_u64(seed);
        let biases = vec![self.longitudinal; self.nvars];
        let mut graph = if let Some(s) = initial_state {
            GraphState::new_with_state_and_rng(s, &self.edges, &biases, rng)
        } else {
            GraphState::new(&self.edges, &biases, rng)
        };
        if let Some(edge_move_importance_sampling) = edge_move_importance_sampling {
            graph.enable_edge_importance_sampling(edge_move_importance_sampling);
        }
        self.graphs.push(graph);
    }

    /// Run a classical monte carlo simulation.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `only_basic_moves`: disallow things other than simple spin flips.
    /// * `edge_move_importance_sampling`: Weight the attempts at edge flips by their energy cost.
    fn run_monte_carlo_sampling(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        nspinupdates: Option<usize>,
        nedgeupdates: Option<usize>,
        nwormupdates: Option<usize>,
        only_basic_moves: Option<bool>,
        thermalization_time: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<bool>>)> {
        let thermalization_time = thermalization_time.unwrap_or(0);

        let sampling_freq = sampling_freq.unwrap_or(1);
        let n_samples = timesteps / sampling_freq;
        let mut energies = Array2::<f64>::default((self.graphs.len(), n_samples));
        let mut states = Array3::<bool>::default((self.graphs.len(), n_samples, self.nvars));

        states
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .zip(energies.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .zip(self.graphs.par_iter_mut())
            .for_each(|((mut s, mut e), gs)| {
                (0..thermalization_time)
                    .try_for_each(|_| {
                        gs.do_time_step(
                            beta,
                            nspinupdates,
                            nedgeupdates,
                            nwormupdates,
                            only_basic_moves,
                        )
                    })
                    .unwrap();
                s.axis_iter_mut(ndarray::Axis(0))
                    .zip(e.iter_mut())
                    .fold(gs, |gs, (mut s, e)| {
                        for _ in 0..sampling_freq {
                            gs.do_time_step(
                                beta,
                                nspinupdates,
                                nedgeupdates,
                                nwormupdates,
                                only_basic_moves,
                            )
                            .unwrap();
                        }
                        s.iter_mut()
                            .zip(gs.state_ref().iter().cloned())
                            .for_each(|(s, b)| *s = b);
                        *e = gs.get_energy();
                        gs
                    });
            });
        let py_energies = energies.into_pyarray(py).to_owned();
        let py_states = states.into_pyarray(py).to_owned();

        Ok((py_energies, py_states))
    }
}
