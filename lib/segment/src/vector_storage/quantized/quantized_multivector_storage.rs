use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;

use common::types::{PointOffsetType, ScoreType};
use quantization::{EncodedVectors, VectorParameters};
use serde::{Deserialize, Serialize};

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::vectors::TypedMultiDenseVectorRef;
use crate::types::{MultiVectorComparator, MultiVectorConfig};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MultivectorOffset {
    pub offset: PointOffsetType,
    pub count: PointOffsetType,
}

pub struct QuantizedMultivectorStorage<TEncodedQuery, QuantizedStorage>
where
    TEncodedQuery: Sized,
    QuantizedStorage: EncodedVectors<TEncodedQuery>,
{
    quantized_storage: QuantizedStorage,
    offsets: Vec<MultivectorOffset>,
    dim: usize,
    multi_vector_config: MultiVectorConfig,
    encoded_query: PhantomData<TEncodedQuery>,
}

impl<TEncodedQuery, QuantizedStorage> QuantizedMultivectorStorage<TEncodedQuery, QuantizedStorage>
where
    TEncodedQuery: Sized,
    QuantizedStorage: EncodedVectors<TEncodedQuery>,
{
    pub fn new(
        dim: usize,
        quantized_storage: QuantizedStorage,
        offsets: Vec<MultivectorOffset>,
        multi_vector_config: MultiVectorConfig,
    ) -> Self {
        Self {
            dim,
            quantized_storage,
            offsets,
            multi_vector_config,
            encoded_query: PhantomData,
        }
    }

    pub fn save_multi(
        &self,
        data_path: &Path,
        meta_path: &Path,
        offsets_path: &Path,
    ) -> OperationResult<()> {
        let offsets_serialized = bincode::serialize(&self.offsets).map_err(|_| {
            OperationError::service_error("Cannot serialize quantized multivector offsets")
        })?;
        let mut file = File::create(offsets_path)?;
        file.write_all(&offsets_serialized)?;
        file.flush()?;

        Ok(self.quantized_storage.save(data_path, meta_path)?)
    }

    pub fn load_multi(
        data_path: &Path,
        meta_path: &Path,
        offsets_path: &Path,
        vector_parameters: &VectorParameters,
        multi_vector_config: &MultiVectorConfig,
    ) -> OperationResult<Self> {
        let mut file = File::open(offsets_path)?;
        let mut offsets_serialized = Vec::new();
        file.read_to_end(&mut offsets_serialized)?;
        let offsets = bincode::deserialize(&offsets_serialized).map_err(|_| {
            OperationError::service_error("Cannot deserialize quantized multivector offsets")
        })?;

        Ok(Self {
            dim: vector_parameters.dim,
            quantized_storage: QuantizedStorage::load(data_path, meta_path, vector_parameters)?,
            offsets,
            multi_vector_config: *multi_vector_config,
            encoded_query: PhantomData,
        })
    }

    fn score_point_max_similarity(&self, query: &Vec<TEncodedQuery>, vector_index: u32) -> f32 {
        let vectors_count = self.offsets[vector_index as usize].count;
        let vectors_offset = self.offsets[vector_index as usize].offset;
        let mut sum = 0.0;
        for inner_query in query {
            let mut max_sim = ScoreType::NEG_INFINITY;
            // manual `max_by` for performance
            for i in 0..vectors_count {
                let sim = self
                    .quantized_storage
                    .score_point(inner_query, vectors_offset + i);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            // sum of max similarity
            sum += max_sim;
        }
        sum
    }

    fn score_internal_max_similarity(&self, vector_a_index: u32, vector_b_index: u32) -> f32 {
        let vector_a_count = self.offsets[vector_a_index as usize].count;
        let vector_b_count = self.offsets[vector_b_index as usize].count;
        let vector_a_offset = self.offsets[vector_a_index as usize].offset;
        let vector_b_offset = self.offsets[vector_b_index as usize].offset;
        let mut sum = 0.0;
        for a in 0..vector_a_count {
            let mut max_sim = ScoreType::NEG_INFINITY;
            // manual `max_by` for performance
            for b in 0..vector_b_count {
                let sim = self
                    .quantized_storage
                    .score_internal(vector_a_offset + a, vector_b_offset + b);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            // sum of max similarity
            sum += max_sim;
        }
        sum
    }
}

impl<TEncodedQuery, QuantizedStorage> EncodedVectors<Vec<TEncodedQuery>>
    for QuantizedMultivectorStorage<TEncodedQuery, QuantizedStorage>
where
    TEncodedQuery: Sized,
    QuantizedStorage: EncodedVectors<TEncodedQuery>,
{
    // TODO(colbert): refactor `EncodedVectors` to support multi vector storage after quantization migration
    fn save(&self, _data_path: &Path, _meta_path: &Path) -> std::io::Result<()> {
        unreachable!("multivector quantized storage should be saved using `self.save_multi` method")
    }

    // TODO(colbert): refactor `EncodedVectors` to support multi vector storage after quantization migration
    fn load(
        _data_path: &Path,
        _meta_path: &Path,
        _vector_parameters: &quantization::VectorParameters,
    ) -> std::io::Result<Self> {
        unreachable!(
            "multivector quantized storage should be loaded using `self.load_multi` method"
        )
    }

    fn encode_query(&self, query: &[f32]) -> Vec<TEncodedQuery> {
        let multi_vector = TypedMultiDenseVectorRef {
            dim: self.dim,
            flattened_vectors: query,
        };
        multi_vector
            .multi_vectors()
            .map(|inner_vector| self.quantized_storage.encode_query(inner_vector))
            .collect()
    }

    fn score_point(&self, query: &Vec<TEncodedQuery>, i: u32) -> f32 {
        match self.multi_vector_config.comparator {
            MultiVectorComparator::MaxSim => self.score_point_max_similarity(query, i),
        }
    }

    fn score_internal(&self, i: u32, j: u32) -> f32 {
        match self.multi_vector_config.comparator {
            MultiVectorComparator::MaxSim => self.score_internal_max_similarity(i, j),
        }
    }
}
