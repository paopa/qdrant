use crate::settings::Settings;
use api::grpc::models::PyroscopeConfig;
use pyroscope::{pyroscope::PyroscopeAgentRunning, PyroscopeAgent};
use pyroscope_pprofrs::{pprof_backend, PprofConfig};

use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct PyroscopeState {
    pub config: Arc<Mutex<PyroscopeConfig>>,
    pub agent: Arc<Mutex<Option<PyroscopeAgent<PyroscopeAgentRunning>>>>,
}

impl PyroscopeState {
    pub fn build_agent(config: &PyroscopeConfig) -> PyroscopeAgent<PyroscopeAgentRunning> {
        let pprof_config = PprofConfig::new().sample_rate(config.sampling_rate.unwrap_or(100));
        let backend_impl = pprof_backend(pprof_config);

        log::info!("Starting agent with identifier {}", &config.identifier);
        let agent = PyroscopeAgent::builder(config.url.to_string(), "qdrant".to_string())
            .backend(backend_impl)
            .tags([("app", "Qdrant"), ("identifier", &config.identifier)].to_vec())
            .build()
            .expect("Couldn't build pyroscope agent");

        agent.start().unwrap()
    }

    pub fn from_config(config: &PyroscopeConfig) -> Self {
        PyroscopeState {
            config: Arc::new(Mutex::new(config.clone())),
            agent: Arc::new(Mutex::new(Some(PyroscopeState::build_agent(config)))),
        }
    }

    pub fn from_settings(settings: &Settings) -> Option<Self> {
        if let Some(debug_config) = settings.debug.clone() {
            Some(PyroscopeState::from_config(&debug_config.pyroscope))
        } else {
            None
        }
    }
}

impl Drop for PyroscopeState {
    fn drop(&mut self) {
        let mut agent_guard = self.agent.lock().unwrap();
        if let Some(running_agent) = agent_guard.take() {
            let ready_agent = running_agent.stop().unwrap();
            ready_agent.shutdown();
        }
    }
}
