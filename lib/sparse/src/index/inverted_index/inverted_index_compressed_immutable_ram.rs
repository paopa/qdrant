use std::path::Path;

use common::types::PointOffsetType;

use super::inverted_index_mmap::InvertedIndexMmap;
use super::inverted_index_ram::InvertedIndexRam;
use super::InvertedIndex;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimOffset};
use crate::index::compressed_posting_list::{
    CompressedPostingBuilder, CompressedPostingList, CompressedPostingListIterator,
};
use crate::index::posting_list_common::PostingListIter as _;

/// A wrapper around [`InvertedIndexRam`].
/// Will be replaced with the new compressed implementation eventually.
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedIndexImmutableRam {
    postings: Vec<CompressedPostingList>,
}

impl InvertedIndex for InvertedIndexImmutableRam {
    type Iter<'a> = CompressedPostingListIterator<'a>;

    fn open(_path: &Path) -> std::io::Result<Self> {
        todo!()
    }

    fn save(&self, _path: &Path) -> std::io::Result<()> {
        todo!()
    }

    fn get(&self, id: &DimId) -> Option<Self::Iter<'_>> {
        self.postings
            .get(*id as usize)
            .map(|posting_list| posting_list.iter())
    }

    fn len(&self) -> usize {
        self.postings.len()
    }

    fn posting_list_len(&self, id: &DimOffset) -> Option<usize> {
        self.get(id).map(|posting_list| posting_list.len_to_end())
    }

    fn files(path: &Path) -> Vec<std::path::PathBuf> {
        InvertedIndexMmap::files(path)
    }

    fn upsert(&mut self, _id: PointOffsetType, _vector: RemappedSparseVector) {
        panic!("Cannot upsert into a read-only RAM inverted index")
    }

    fn from_ram_index<P: AsRef<Path>>(
        ram_index: InvertedIndexRam,
        _path: P,
    ) -> std::io::Result<Self> {
        let mut postings = Vec::with_capacity(ram_index.postings.len());
        for old_posting_list in ram_index.postings {
            let mut new_posting_list = CompressedPostingBuilder::new();
            for elem in old_posting_list.elements {
                new_posting_list.add(elem.record_id, elem.weight);
            }
            postings.push(new_posting_list.build());
        }
        Ok(InvertedIndexImmutableRam { postings })
    }

    fn vector_count(&self) -> usize {
        todo!()
    }

    fn max_index(&self) -> Option<DimOffset> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::Builder;

    use super::*;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn inverted_index_ram_save_load() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, vec![(1, 10.0), (2, 10.0), (3, 10.0)].try_into().unwrap());
        builder.add(2, vec![(1, 20.0), (2, 20.0), (3, 20.0)].try_into().unwrap());
        builder.add(3, vec![(1, 30.0), (2, 30.0), (3, 30.0)].try_into().unwrap());
        let inverted_index_ram = builder.build();

        let tmp_dir_path = Builder::new().prefix("test_index_dir").tempdir().unwrap();
        let inverted_index_immutable_ram =
            InvertedIndexImmutableRam::from_ram_index(inverted_index_ram, tmp_dir_path.path())
                .unwrap();
        inverted_index_immutable_ram
            .save(tmp_dir_path.path())
            .unwrap();

        let loaded_inverted_index = InvertedIndexImmutableRam::open(tmp_dir_path.path()).unwrap();
        assert_eq!(inverted_index_immutable_ram, loaded_inverted_index);
    }
}
