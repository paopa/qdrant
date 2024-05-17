use segment::types::ShardKey;
use serde::{Deserialize, Serialize};

use crate::shards::shard::{PeerId, ShardId};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct State {
    pub peer_id: PeerId,
    pub shard_id: ShardId,
    pub shard_key: Option<ShardKey>,
}

impl State {
    #[allow(dead_code)]
    pub fn new(peer_id: PeerId, shard_id: ShardId, shard_key: Option<ShardKey>) -> Self {
        Self {
            peer_id,
            shard_id,
            shard_key,
        }
    }
}
