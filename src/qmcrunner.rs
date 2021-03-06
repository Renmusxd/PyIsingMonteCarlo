use itertools::Itertools;
use ndarray::{Array, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::PyErr;
use qmc::classical::graph::make_random_spin_state;
use qmc::sse::fast_op_alloc::{DefaultFastOpAllocator, SwitchableFastOpAllocator};
use qmc::sse::fast_ops::{FastOp, FastOpsTemplate};
use qmc::sse::*;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::min;

type SwitchFastOp = FastOpsTemplate<FastOp, SwitchableFastOpAllocator<DefaultFastOpAllocator>>;
type SwitchQmc = Qmc<SmallRng, SwitchFastOp>;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
#[derive(Clone)]
pub struct QmcRunner {
    nvars: usize,
    do_loop_updates: bool,
    do_heatbath: bool,
    interactions: Vec<(Vec<f64>, Vec<usize>)>,
    qmc: Vec<SwitchQmc>,
    rng: SmallRng,
    use_allocator: bool,
}

#[pymethods]
impl QmcRunner {
    /// Construct a new instance with `num_experiments` qmc instances.
    #[new]
    fn new(
        nvars: usize,
        num_experiments: usize,
        seed: Option<u64>,
        use_allocator: Option<bool>,
        do_loop_updates: Option<bool>,
        do_heatbath_updates: Option<bool>,
    ) -> Self {
        let rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let do_loop_updates = do_loop_updates.unwrap_or(false);
        let do_heatbath = do_heatbath_updates.unwrap_or(false);
        let use_allocator = use_allocator.unwrap_or(true);
        let mut s = Self {
            nvars,
            do_loop_updates,
            do_heatbath,
            interactions: Vec::default(),
            qmc: Default::default(),
            rng,
            use_allocator,
        };
        (0..num_experiments).for_each(|_| s.add_qmc(None));
        s
    }

    /// Add a new experiment.
    fn add_qmc(&mut self, use_allocator: Option<bool>) {
        let use_allocator = use_allocator.unwrap_or(self.use_allocator);
        let seed = self.rng.gen();
        let mut rng = SmallRng::seed_from_u64(seed);
        let state = make_random_spin_state(self.nvars, &mut rng);
        let mut qmc = SwitchQmc::new_with_state_with_manager_hook(
            self.nvars,
            rng,
            state,
            self.do_loop_updates,
            |nvars| {
                let alloc = if use_allocator {
                    Some(DefaultFastOpAllocator::default())
                } else {
                    None
                };
                let alloc = SwitchableFastOpAllocator::new(alloc);
                SwitchFastOp::new_from_nvars_and_nbonds_and_alloc(nvars, None, alloc)
            },
        );

        self.interactions
            .iter()
            .for_each(|(mat, vars)| qmc.make_interaction(mat.clone(), vars.clone()).unwrap());
        qmc.set_do_heatbath(self.do_heatbath);
        self.qmc.push(qmc);
    }

    /// Add an interaction to all experiments.
    fn add_interaction(&mut self, mat: Vec<f64>, vars: Vec<usize>) -> PyResult<()> {
        self.qmc
            .iter_mut()
            .try_for_each(|qmc| qmc.make_interaction(mat.clone(), vars.clone()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, String>)?;
        self.interactions.push((mat, vars));
        Ok(())
    }

    /// Add an interaction to all experiments.
    fn add_interaction_and_offset(&mut self, mat: Vec<f64>, vars: Vec<usize>) -> PyResult<()> {
        self.qmc
            .iter_mut()
            .try_for_each(|qmc| qmc.make_interaction_and_offset(mat.clone(), vars.clone()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, String>)?;
        self.interactions.push((mat, vars));
        Ok(())
    }

    /// Add a diagonal interaction to all experiments.
    fn add_diagonal_interaction(&mut self, mat: Vec<f64>, vars: Vec<usize>) -> PyResult<()> {
        self.qmc
            .iter_mut()
            .try_for_each(|qmc| qmc.make_diagonal_interaction(mat.clone(), vars.clone()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, String>)?;
        self.interactions.push((mat, vars));
        Ok(())
    }

    /// Add a diagonal interaction to all experiments.
    fn add_diagonal_interaction_and_offset(
        &mut self,
        mat: Vec<f64>,
        vars: Vec<usize>,
    ) -> PyResult<()> {
        self.qmc
            .iter_mut()
            .try_for_each(|qmc| qmc.make_diagonal_interaction_and_offset(mat.clone(), vars.clone()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, String>)?;
        self.interactions.push((mat, vars));
        Ok(())
    }

    /// Set whether the experiments are allowed to do heatbath updates.
    fn set_do_heatbath(&mut self, do_heatbath: bool) {
        self.do_heatbath = do_heatbath;
        self.qmc
            .iter_mut()
            .for_each(|qmc| qmc.set_do_heatbath(do_heatbath))
    }

    /// Set whether the experiments are allowed to do loop updates.
    fn set_do_loop_updates(&mut self, do_loop_updates: bool) {
        self.do_loop_updates = do_loop_updates;
        self.qmc
            .iter_mut()
            .for_each(|qmc| qmc.set_do_loop_updates(do_loop_updates))
    }

    /// Run a quantum monte carlo simulation, sample the state at each `sampling_freq`, returns that
    /// array plus the average energy.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to sample.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_sampling(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> (Py<PyArray1<f64>>, Py<PyArray3<bool>>) {
        let sampling_wait_buffer = sampling_wait_buffer.map(|wait| min(wait, timesteps));

        let mut energies = Array::default((self.qmc.len(),));
        let mut states = Array3::<bool>::default((
            self.qmc.len(),
            timesteps / sampling_freq.unwrap_or(1),
            self.nvars,
        ));

        let nvars = self.nvars;
        self.qmc
            .par_iter_mut()
            .zip(states.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .zip(energies.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .for_each(|((qmc_graph, mut s), mut e)| {
                if let Some(wait) = sampling_wait_buffer {
                    qmc_graph.timesteps(wait, beta);
                };

                let energy = qmc_graph.timesteps_sample_iter_zip(
                    timesteps,
                    beta,
                    sampling_freq,
                    s.iter_mut().chunks(nvars).into_iter(),
                    |buf, s| buf.zip(s.iter()).for_each(|(b, s)| *b = *s),
                );
                e.fill(energy);
            });
        let py_energies = energies.into_pyarray(py).to_owned();
        let py_states = states.into_pyarray(py).to_owned();

        (py_energies, py_states)
    }

    /// Run a quantum monte carlo simulation, sample the bonds at each `sampling_freq`, returns the
    /// average number of each.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to sample.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_bond_sampling(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> Py<PyArray3<usize>> {
        let sampling_wait_buffer = sampling_wait_buffer.map(|wait| min(wait, timesteps));

        let nbonds = self.interactions.len();
        let mut bonds = Array::default((
            self.qmc.len(),
            timesteps / sampling_freq.unwrap_or(1),
            nbonds,
        ));

        self.qmc
            .par_iter_mut()
            .zip(bonds.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .for_each(|(qmc_graph, mut bs)| {
                if let Some(wait) = sampling_wait_buffer {
                    qmc_graph.timesteps(wait, beta);
                };

                qmc_graph.timesteps_iter_zip_with_self(
                    timesteps,
                    beta,
                    sampling_freq,
                    bs.iter_mut().chunks(nbonds).into_iter(),
                    |buf, s| {
                        buf.zip(0..nbonds)
                            .for_each(|(bb, b)| *bb = s.get_bond_count(b))
                    },
                );
            });
        bonds.into_pyarray(py).to_owned()
    }

    /// Run a quantum monte carlo simulation, get the variable's autocorrelation across time for
    /// each experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_variable_autocorrelation(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> Py<PyArray2<f64>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        let mut corrs = Array::default((self.qmc.len(), timesteps));
        self.qmc
            .par_iter_mut()
            .zip(corrs.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .for_each(|(qmc_graph, mut corrs)| {
                if sampling_wait_buffer > 0 {
                    qmc_graph.timesteps(sampling_wait_buffer, beta);
                }

                let auto =
                    qmc_graph.calculate_variable_autocorrelation(timesteps, beta, sampling_freq);
                corrs
                    .iter_mut()
                    .zip(auto.into_iter())
                    .for_each(|(c, v)| *c = v);
            });
        corrs.into_pyarray(py).to_owned()
    }

    /// Run a quantum monte carlo simulation, get the variable's autocorrelation across time for
    /// each experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `spin_products`: List of list of variables to take products of.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_spin_product_autocorrelation(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        spin_products: Vec<Vec<usize>>,
        sampling_wait_buffer: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> Py<PyArray2<f64>> {
        let spin_refs = spin_products
            .iter()
            .map(|p| p.as_slice())
            .collect::<Vec<_>>();
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        let mut corrs = Array::default((self.qmc.len(), timesteps));
        self.qmc
            .par_iter_mut()
            .zip(corrs.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .for_each(|(qmc_graph, mut corrs)| {
                if sampling_wait_buffer > 0 {
                    qmc_graph.timesteps(sampling_wait_buffer, beta);
                }

                let auto = qmc_graph.calculate_spin_product_autocorrelation(
                    timesteps,
                    beta,
                    &spin_refs,
                    sampling_freq,
                );
                corrs
                    .iter_mut()
                    .zip(auto.into_iter())
                    .for_each(|(c, v)| *c = v);
            });
        corrs.into_pyarray(py).to_owned()
    }

    /// Run a quantum monte carlo simulation, get the bond's autocorrelation across time for each
    /// experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_bond_autocorrelation(
        &mut self,
        py: Python,
        beta: f64,
        timesteps: usize,
        sampling_wait_buffer: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> Py<PyArray2<f64>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        let mut corrs = Array::default((self.qmc.len(), timesteps));
        self.qmc
            .par_iter_mut()
            .zip(corrs.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
            .for_each(|(qmc_graph, mut corrs)| {
                if sampling_wait_buffer > 0 {
                    qmc_graph.timesteps(sampling_wait_buffer, beta);
                }

                let auto = qmc_graph.calculate_bond_autocorrelation(timesteps, beta, sampling_freq);
                corrs
                    .iter_mut()
                    .zip(auto.into_iter())
                    .for_each(|(c, v)| *c = v);
            });
        corrs.into_pyarray(py).to_owned()
    }

    /// Get the energy offset calculated from the set of interactions.
    pub fn get_offset(&self) -> f64 {
        if !self.qmc.is_empty() {
            self.qmc[0].get_offset()
        } else {
            0.0
        }
    }

    /// Get the imaginary time states for the selected graph.
    fn get_graph_itime(&self, py: Python, g: usize) -> PyResult<Py<PyArray2<bool>>> {
        let graph = self.qmc.get(g);
        if let Some(g) = graph {
            let mut states = Array2::<bool>::default((g.get_cutoff(), self.nvars));
            let axis_iter = states.axis_iter_mut(ndarray::Axis(0));

            g.imaginary_time_fold(
                |mut it, s| {
                    let mut row = it.next().unwrap();
                    row.iter_mut().zip(s.iter().cloned()).for_each(|(b, sb)| {
                        *b = sb;
                    });
                    it
                },
                axis_iter,
            );

            let py_states = states.into_pyarray(py).to_owned();
            Ok(py_states)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, String>(
                format!("Attempted to get graph {} of {}", g, self.qmc.len()),
            ))
        }
    }

    /// Clone the given graphs.
    fn clone(&self) -> Self {
        <Self as Clone>::clone(self)
    }
}
