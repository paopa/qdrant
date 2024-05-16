use std::ops::ControlFlow;

use common::types::PointOffsetType;

use crate::common::types::DimWeight;

pub const DEFAULT_MAX_NEXT_WEIGHT: DimWeight = f32::NEG_INFINITY;

#[derive(Debug, Clone, PartialEq)]
pub struct PostingElement {
    /// Record ID
    pub record_id: PointOffsetType,
    /// Weight of the record in the dimension
    pub weight: DimWeight,
    /// Max weight of the next elements in the posting list.
    pub max_next_weight: DimWeight,
}

impl PostingElement {
    /// Initialize negative infinity as max_next_weight.
    /// Needs to be updated at insertion time.
    pub(crate) fn new(record_id: PointOffsetType, weight: DimWeight) -> PostingElement {
        PostingElement {
            record_id,
            weight,
            max_next_weight: DEFAULT_MAX_NEXT_WEIGHT,
        }
    }
}

pub trait PostingListIter {
    fn peek(&mut self) -> Option<PostingElement>;

    fn last(&self) -> Option<PostingElement>;

    fn skip_to(&mut self, record_id: PointOffsetType) -> Option<PostingElement>;

    fn skip_to_end(&mut self);

    fn len_to_end(&self) -> usize;

    fn current_index(&self) -> usize;

    fn try_for_each<F, R>(&mut self, f: F) -> ControlFlow<R>
    where
        F: FnMut(PostingElement) -> ControlFlow<R>;
}
