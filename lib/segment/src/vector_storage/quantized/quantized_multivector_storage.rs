use std::path::Path;

use crate::common::operation_error::OperationResult;

pub struct QuantizedMultivectorStorage<QuantizedStorage> {
    quantized_storage: QuantizedStorage,
}

impl<QuantizedStorage> QuantizedMultivectorStorage<QuantizedStorage> {
    pub fn new(quantized_storage: QuantizedStorage) -> Self {
        QuantizedMultivectorStorage { quantized_storage }
    }

    pub fn save(&self, _data_path: &Path, _meta_path: &Path) -> OperationResult<()> {
        Ok(())
    }
}
