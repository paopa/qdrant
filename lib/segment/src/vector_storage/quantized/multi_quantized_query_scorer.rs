use std::borrow::Cow;
use std::marker::PhantomData;

use common::types::{PointOffsetType, ScoreType};
use itertools::Itertools;

use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::MultiDenseVector;
use crate::spaces::metric::Metric;
use crate::types::QuantizationConfig;
use crate::vector_storage::query_scorer::QueryScorer;

pub struct MultiQuantizedQueryScorer<'a, TElement, TMetric, TEncodedQuery, TEncodedVectors>
where
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TEncodedVectors: quantization::EncodedVectors<TEncodedQuery>,
{
    query: TEncodedQuery,
    quantized_data: &'a TEncodedVectors,
    metric: PhantomData<TMetric>,
    element: PhantomData<TElement>,
}

impl<'a, TElement, TMetric, TEncodedQuery, TEncodedVectors>
    MultiQuantizedQueryScorer<'a, TElement, TMetric, TEncodedQuery, TEncodedVectors>
where
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TEncodedVectors: quantization::EncodedVectors<TEncodedQuery>,
{
    pub fn new(
        raw_query: MultiDenseVector,
        quantized_data: &'a TEncodedVectors,
        quantization_config: &QuantizationConfig,
    ) -> Self {
        let slices = raw_query.multi_vectors();
        let query = slices
            .into_iter()
            .flat_map(|inner_vector| {
                let inner_preprocessed = TMetric::preprocess(inner_vector.to_vec());
                let inner_converted =
                    TElement::slice_from_float_cow(Cow::Owned(inner_preprocessed));
                let inner_prequantized = TElement::quantization_preprocess(
                    quantization_config,
                    TMetric::distance(),
                    inner_converted.as_ref(),
                )
                .into_owned();
                inner_prequantized.into_iter()
            })
            .collect_vec();

        let query = quantized_data.encode_query(&query);

        Self {
            query,
            quantized_data,
            metric: PhantomData,
            element: PhantomData,
        }
    }
}

impl<TElement, TMetric, TEncodedQuery, TEncodedVectors> QueryScorer<[TElement]>
    for MultiQuantizedQueryScorer<'_, TElement, TMetric, TEncodedQuery, TEncodedVectors>
where
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TEncodedVectors: quantization::EncodedVectors<TEncodedQuery>,
{
    fn score_stored(&self, idx: PointOffsetType) -> ScoreType {
        self.quantized_data.score_point(&self.query, idx)
    }

    fn score(&self, _: &[TElement]) -> ScoreType {
        unimplemented!("This method is not expected to be called for quantized scorer");
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.quantized_data.score_internal(point_a, point_b)
    }
}
