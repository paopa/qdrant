use std::path::Path;

use common::types::PointOffsetType;
use quantization::EncodedVectors;

use crate::common::operation_error::OperationResult;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct MultivectorOffset {
    offset: PointOffsetType,
    count: PointOffsetType,
}

pub struct QuantizedMultivectorStorage<QuantizedStorage> {
    _quantized_storage: QuantizedStorage,
    _offsets: Vec<MultivectorOffset>,
}

impl<QuantizedStorage> QuantizedMultivectorStorage<QuantizedStorage> {
    pub fn save(&self, _data_path: &Path, _meta_path: &Path) -> OperationResult<()> {
        Ok(())
    }
}

impl<TEncodedQuery, QuantizedStorage> EncodedVectors<Vec<TEncodedQuery>>
    for QuantizedMultivectorStorage<QuantizedStorage>
where
    TEncodedQuery: Sized,
    QuantizedStorage: EncodedVectors<TEncodedQuery>,
{
    fn save(&self, _data_path: &Path, _meta_path: &Path) -> std::io::Result<()> {
        todo!()
    }

    fn load(
        _data_path: &Path,
        _meta_path: &Path,
        _vector_parameters: &quantization::VectorParameters,
    ) -> std::io::Result<Self> {
        todo!()
    }

    fn encode_query(&self, _query: &[f32]) -> Vec<TEncodedQuery> {
        todo!()
    }

    fn score_point(&self, _query: &Vec<TEncodedQuery>, _i: u32) -> f32 {
        todo!()
    }

    fn score_internal(&self, _i: u32, _j: u32) -> f32 {
        todo!()
    }
}
