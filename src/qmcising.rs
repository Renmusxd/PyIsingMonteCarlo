use itertools::Itertools;
use ndarray::{Array, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::PyErr;
use qmc::classical::graph::Edge;
use qmc::sse::fast_op_alloc::{DefaultFastOpAllocator, SwitchableFastOpAllocator};
use qmc::sse::fast_ops::{FastOp, FastOpsTemplate};
use qmc::sse::qmc_ising::serialization::SerializeQmcGraph;
use qmc::sse::*;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::ser::Serialize;
use serde_cbor::ser::IoWrite;
use std::cmp::{max, min};
use std::fs::File;

type SwitchFastOp = FastOpsTemplate<FastOp, SwitchableFastOpAllocator<DefaultFastOpAllocator>>;
type SwitchQmc = QmcIsingGraph<SmallRng, SwitchFastOp>;

type FileType = (
    usize,
    Vec<(Edge, f64)>,
    f64,
    f64,
    bool,
    bool,
    Option<u64>,
    bool,
    Vec<SerializeQmcGraph<SwitchFastOp>>,
);

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
#[derive(Clone)]
pub struct QmcIsing {
    nvars: usize,
    edges: Vec<((usize, usize), f64)>,
    transverse: f64,
    longitudinal: f64,
    enable_heatbath: bool,
    enable_rvb: bool,
    qmc: Vec<SwitchQmc>,
    rng: SmallRng,
    use_allocator: bool,
    seed: Option<u64>,
}

#[pymethods]
impl QmcIsing {
    /// Construct a new instance.
    #[new]
    fn new(
        edges: Vec<((usize, usize), f64)>,
        transverse: f64,
        longitudinal: Option<f64>,
        num_experiments: Option<usize>,
        seed: Option<u64>,
        use_allocator: Option<bool>,
        do_heatbath_updates: Option<bool>,
        do_rvb_updates: Option<bool>,
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
        let do_heatbath = do_heatbath_updates.unwrap_or(false);
        let do_rvb_updates = do_rvb_updates.unwrap_or(false);
        let use_allocator = use_allocator.unwrap_or(true);
        let longitudinal = longitudinal.unwrap_or(0.0);
        let num_experiments = num_experiments.unwrap_or(1);
        let mut s = Self {
            nvars,
            edges,
            transverse,
            longitudinal,
            enable_heatbath: do_heatbath,
            enable_rvb: do_rvb_updates,
            qmc: Default::default(),
            rng,
            use_allocator,
            seed,
        };
        (0..num_experiments).for_each(|_| s.add_qmc(None));
        s
    }

    /// Add a new experiment.
    fn add_qmc(&mut self, use_allocator: Option<bool>) {
        let use_allocator = use_allocator.unwrap_or(self.use_allocator);
        let seed = self.rng.gen();
        let rng = SmallRng::seed_from_u64(seed);
        let mut qmc = SwitchQmc::new_with_rng_with_manager_hook(
            self.edges.clone(),
            self.transverse,
            self.longitudinal,
            self.nvars,
            rng,
            None,
            |nvars, nbonds| {
                let alloc = if use_allocator {
                    Some(DefaultFastOpAllocator::default())
                } else {
                    None
                };
                let alloc = SwitchableFastOpAllocator::new(alloc);
                SwitchFastOp::new_from_nvars_and_nbonds_and_alloc(nvars, Some(nbonds), alloc)
            },
        );
        qmc.set_enable_heatbath(self.enable_heatbath);
        qmc.set_run_rvb(self.enable_rvb);
        self.qmc.push(qmc);
    }

    /// Set whether the experiments are allowed to do heatbath updates.
    fn set_enable_heatbath(&mut self, enable_heatbath: bool) {
        self.enable_heatbath = enable_heatbath;
        self.qmc
            .iter_mut()
            .for_each(|qmc| qmc.set_enable_heatbath(enable_heatbath))
    }
    /// Set whether the experiments are allowed to do rvb updates.
    fn set_enable_rvb(&mut self, enable_rvb: bool) {
        self.enable_rvb = enable_rvb;
        self.qmc
            .iter_mut()
            .for_each(|qmc| qmc.set_run_rvb(enable_rvb))
    }

    /// Run `timesteps` at `beta` in parallel for each graph.
    fn run_qmc(&mut self, beta: f64, timesteps: usize) {
        self.qmc.iter_mut().for_each(|qmc| {
            qmc.timesteps(timesteps, beta);
        });
    }

    /// Run at `beta` in parallel for each graph.
    fn run_diagonal(&mut self, beta: f64, timesteps: Option<usize>) {
        let timesteps = timesteps.unwrap_or(1);
        self.qmc.par_iter_mut().for_each(|qmc| {
            (0..timesteps).for_each(|_| qmc.single_diagonal_step(beta));
        });
    }

    /// Run in parallel for each graph.
    fn run_cluster(&mut self, timesteps: Option<usize>) {
        let timesteps = timesteps.unwrap_or(1);
        self.qmc.par_iter_mut().for_each(|qmc| {
            (0..timesteps).for_each(|_| qmc.single_cluster_step());
        });
    }

    /// Run in parallel for each graph.
    fn run_rvb(&mut self, timesteps: Option<usize>) {
        let timesteps = timesteps.unwrap_or(1);
        self.qmc.par_iter_mut().for_each(|qmc| {
            (0..timesteps).for_each(|_| qmc.single_rvb_step());
        });
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

        let nbonds = self.edges.len();
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

    /// Save graphs to a filepath. Does not save state of RNG.
    fn save_to_file(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)?;
        let qmc: Vec<SerializeQmcGraph<SwitchFastOp>> =
            self.qmc.iter().cloned().map(|q| q.into()).collect();
        let to_write: FileType = (
            self.nvars,
            self.edges.clone(),
            self.transverse,
            self.longitudinal,
            self.enable_heatbath,
            self.enable_rvb,
            self.seed,
            self.use_allocator,
            qmc,
        );
        to_write
            .serialize(&mut serde_cbor::Serializer::new(&mut IoWrite::new(f)).packed_format())
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyIOError, String>(err.to_string()))
    }

    /// Load graphs from a filepath. Does not load state of RNG. In order to create repeatable
    /// calculations reseed with a new value. If no reseed is provided rng is seeded from entropy.
    #[staticmethod]
    fn read_from_file(path: &str, reseed: Option<u64>) -> PyResult<Self> {
        let f = File::open(path)?;
        let (
            nvars,
            edges,
            transverse,
            longitudinal,
            enable_heatbath,
            enable_rvb,
            seed,
            use_allocator,
            qmc,
        ): FileType = serde_cbor::from_reader(f)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyIOError, String>(err.to_string()))?;
        // Do _NOT_ seed rng from saved value since that would repeat previous numbers,
        // not resume where it left off.
        let mut container_rng = if let Some(reseed) = reseed {
            SmallRng::seed_from_u64(reseed)
        } else {
            SmallRng::from_entropy()
        };
        let qmc = qmc
            .into_iter()
            .map(|qmc| {
                let seed = container_rng.gen();
                let rng = SmallRng::seed_from_u64(seed);
                qmc.into_qmc(rng)
            })
            .collect();
        Ok(Self {
            nvars,
            edges,
            transverse,
            longitudinal,
            enable_heatbath,
            enable_rvb,
            qmc,
            seed,
            use_allocator,
            rng: container_rng,
        })
    }
}
