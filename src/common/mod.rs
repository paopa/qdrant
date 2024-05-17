#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod collections;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod error_reporting;
#[allow(dead_code)]
pub mod health;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod helpers;
pub mod http_client;
pub mod metrics;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod points;
pub mod snapshots;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod stacktrace;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod telemetry;
pub mod telemetry_ops;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod telemetry_reporting;

pub mod auth;

pub mod strings;

pub mod pyroscope_state;
