use ndarray::{Array, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use qmc::classical::graph::Edge;
use qmc::sse::fast_op_alloc::{DefaultFastOpAllocator, SwitchableFastOpAllocator};
use qmc::sse::fast_ops::{FastOp, FastOpsTemplate};
use qmc::sse::parallel_tempering::*;
use qmc::sse::*;
use rand::prelude::*;
use rayon::prelude::*;
use serde::ser::Serialize;
use serde_cbor::ser::IoWrite;
use std::cmp::{max, min};
use std::fs::File;

type SwitchFastOp = FastOpsTemplate<FastOp, SwitchableFastOpAllocator<DefaultFastOpAllocator>>;
type SwitchQmc = QmcIsingGraph<SmallRng, SwitchFastOp>;
type TempCont = TemperingContainer<SmallRng, SwitchQmc>;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
#[derive(Clone)]
pub struct LatticeTempering {
    nvars: usize,
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
    tempering: TempCont,
    seed: Option<u64>,
    use_allocator: bool,
}

#[pymethods]
impl LatticeTempering {
    #[new]
    fn new(edges: Vec<(Edge, f64)>, seed: Option<u64>, use_allocator: Option<bool>) -> Self {
        let nvars = edges
            .iter()
            .map(|((a, b), _)| max(*a, *b))
            .max()
            .map(|x| x + 1)
            .unwrap();
        let cutoff = nvars;
        let rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let use_allocator = use_allocator.unwrap_or(true);
        let tempering = TempCont::new(rng);
        Self {
            nvars,
            edges,
            cutoff,
            tempering,
            seed,
            use_allocator,
        }
    }

    /// Add a graph to be run with field `transverse` and `longitudinal` at `beta`.
    fn add_graph(
        &mut self,
        transverse: f64,
        longitudinal: f64,
        beta: f64,
        edges: Option<Vec<(Edge, f64)>>,
        enable_rvb_update: Option<bool>,
        enable_heatbath_update: Option<bool>,
        seed: Option<u64>,
        use_allocator: Option<bool>,
    ) -> PyResult<()> {
        let edges = match (edges, &self.edges) {
            (Some(edges), _) => edges,
            (None, edges) => edges.clone(),
        };
        let seed = seed.unwrap_or_else(|| self.tempering.rng_mut().gen());
        let use_allocator = use_allocator.unwrap_or(self.use_allocator);
        let rng = SmallRng::seed_from_u64(seed);
        let rvb = enable_rvb_update.unwrap_or(false);
        let heatbath = enable_heatbath_update.unwrap_or(false);
        // Add a hook to insert own allocator.
        let mut qmc = SwitchQmc::new_with_rng_with_manager_hook(
            edges,
            transverse,
            longitudinal,
            self.cutoff,
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
        qmc.set_run_rvb(rvb);
        qmc.set_enable_heatbath(heatbath);
        self.tempering
            .add_qmc_stepper(qmc, beta)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, String>)
    }

    fn get_num_graphs(&self) -> usize {
        self.tempering.num_graphs()
    }

    fn get_graph_itime(&self, py: Python, g: usize) -> PyResult<Py<PyArray2<bool>>> {
        let graph = self.tempering.graph_ref().get(g);
        if let Some((g, _)) = graph {
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
                format!(
                    "Attempted to get graph {} of {}",
                    g,
                    self.tempering.num_graphs()
                ),
            ))
        }
    }

    /// Run `t` qmc timesteps on each graph.
    fn qmc_timesteps(&mut self, t: usize) {
        self.tempering.parallel_timesteps(t)
    }

    /// Run Qmc timesteps and sample every `sampling_freq` steps, take a parallel tempering step
    /// every `replica_swap_freq` steps if nonzero.
    fn qmc_timesteps_sample(
        &mut self,
        py: Python,
        timesteps: usize,
        replica_swap_freq: Option<usize>,
        sampling_freq: Option<usize>,
    ) -> (Py<PyArray3<bool>>, Py<PyArray1<f64>>) {
        let sampling_freq = sampling_freq.unwrap_or(1);
        let replica_swap_freq = replica_swap_freq.unwrap_or(1);
        let mut states = Array3::<bool>::default((
            self.tempering.num_graphs(),
            timesteps / sampling_freq,
            self.nvars,
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

            if time_to_swap == 0 && replica_swap_freq > 0 {
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
        (py_states, py_energies)
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
    ) -> Py<PyArray2<f64>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        if sampling_wait_buffer > 0 {
            self.tempering.parallel_timesteps(sampling_wait_buffer);
        }

        let autos = self.tempering.calculate_variable_autocorrelation(
            timesteps,
            replica_swap_freq,
            sampling_freq,
        );
        let mut corrs = Array::default((self.tempering.num_graphs(), timesteps));
        corrs
            .axis_iter_mut(ndarray::Axis(0))
            .zip(autos.into_iter())
            .for_each(|(mut s, auto)| s.iter_mut().zip(auto.into_iter()).for_each(|(a, b)| *a = b));

        corrs.into_pyarray(py).to_owned()
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
    ) -> Py<PyArray2<f64>> {
        let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
        if sampling_wait_buffer > 0 {
            self.tempering.parallel_timesteps(sampling_wait_buffer);
        }

        let autos = self.tempering.calculate_bond_autocorrelation(
            timesteps,
            replica_swap_freq,
            sampling_freq,
        );
        let mut corrs = Array::default((self.tempering.num_graphs(), timesteps));
        corrs
            .axis_iter_mut(ndarray::Axis(0))
            .zip(autos.into_iter())
            .for_each(|(mut s, auto)| s.iter_mut().zip(auto.into_iter()).for_each(|(a, b)| *a = b));

        corrs.into_pyarray(py).to_owned()
    }

    /// Get the total number of tempering swaps which have occurred.
    fn get_total_swaps(&self) -> u64 {
        self.tempering.get_total_swaps()
    }

    /// Clone the given tempering object and all its graphs.
    fn clone(&self) -> Self {
        <Self as Clone>::clone(self)
    }

    /// Save graphs to a filepath. Does not save state of RNG.
    fn save_to_file(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)?;
        let tempering: SerializeTemperingContainer<SwitchFastOp> = self.tempering.clone().into();
        let to_write = (
            self.nvars,
            self.edges.clone(),
            self.cutoff,
            self.seed,
            self.use_allocator,
            tempering,
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
        let (nvars, edges, cutoff, seed, use_allocator, tempering): (
            usize,
            Vec<(Edge, f64)>,
            usize,
            Option<u64>,
            bool,
            SerializeTemperingContainer<SwitchFastOp>,
        ) = serde_cbor::from_reader(f)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyIOError, String>(err.to_string()))?;
        // Do _NOT_ seed rng from saved value since that would repeat previous numbers,
        // not resume where it left off.
        let container_rng = if let Some(seed) = reseed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        Ok(Self {
            nvars,
            edges,
            cutoff,
            tempering: tempering.into_tempering_container_gen_rngs(container_rng),
            seed,
            use_allocator,
        })
    }
}
