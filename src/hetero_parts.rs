// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Versioned heterogeneous placement stored alongside the legacy Boomerang
//! partitions. The legacy payload remains available to V1; V2 consumes the
//! explicit per-wave queues below and never guesses placement from endpoint
//! ordering.

use serde::{Deserialize, Serialize};

use crate::pe::Partition;
use crate::schedule::HeteroSchedule;

pub const GEM_PARTS_V2_MAGIC: u64 = 0x3254_5241_504d_4547; // "GEMPART2" LE
pub const GEM_PARTS_V2_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionWave {
    pub aig_regions: Vec<u32>,
    pub carry4: Vec<u32>,
    pub dsp48e2: Vec<u32>,
    pub srlc32e: Vec<u32>,
}

impl PartitionWave {
    fn all_nodes(&self) -> impl Iterator<Item = u32> + '_ {
        self.aig_regions
            .iter()
            .chain(&self.carry4)
            .chain(&self.dsp48e2)
            .chain(&self.srlc32e)
            .copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeteroPlacementV2 {
    pub num_partitions: u32,
    /// `node_partition[node_id]` is the CUDA block that owns the operation.
    pub node_partition: Vec<u32>,
    /// `[wave][partition]` type-homogeneous queues of schedule node IDs.
    pub waves: Vec<Vec<PartitionWave>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemPartsV2 {
    pub magic: u64,
    pub version: u32,
    pub legacy: Vec<Vec<Partition>>,
    pub hetero: HeteroPlacementV2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartsV2Error {
    BadMagic(u64),
    UnsupportedVersion(u32),
    ZeroPartitions,
    BadWaveCount {
        expected: usize,
        got: usize,
    },
    BadPartitionCount {
        wave: usize,
        expected: usize,
        got: usize,
    },
    BadNode {
        node: usize,
    },
    NodePlacedTwice {
        node: usize,
    },
    MissingNode {
        node: usize,
    },
    PartitionMismatch {
        node: usize,
        expected: u32,
        got: u32,
    },
    WrongWave {
        node: usize,
        expected: usize,
        got: usize,
    },
    WrongKind {
        node: usize,
    },
}

impl HeteroPlacementV2 {
    /// Deterministic balanced placement. A node is assigned to the least-loaded
    /// partition, with ties broken by partition ID. Direct dependencies may
    /// cross blocks; the executor publishes every wave before its grid barrier.
    pub fn build(schedule: &HeteroSchedule, num_partitions: usize) -> Result<Self, PartsV2Error> {
        if num_partitions == 0 {
            return Err(PartsV2Error::ZeroPartitions);
        }
        let mut loads = vec![0usize; num_partitions];
        let mut node_partition = vec![0u32; schedule.nodes.len()];
        let mut waves = vec![vec![PartitionWave::default(); num_partitions]; schedule.waves.len()];

        for (wave_id, wave) in schedule.waves.iter().enumerate() {
            let queues: [(&[usize], u8); 4] = [
                (&wave.aig_regions, 0),
                (&wave.carry4, 1),
                (&wave.dsp48e2, 2),
                (&wave.srlc32e, 3),
            ];
            for (queue, kind) in queues {
                for &node in queue {
                    let part = loads
                        .iter()
                        .enumerate()
                        .min_by_key(|&(part, load)| (*load, part))
                        .unwrap()
                        .0;
                    loads[part] += 1;
                    node_partition[node] = part as u32;
                    let dst = &mut waves[wave_id][part];
                    match kind {
                        0 => dst.aig_regions.push(node as u32),
                        1 => dst.carry4.push(node as u32),
                        2 => dst.dsp48e2.push(node as u32),
                        _ => dst.srlc32e.push(node as u32),
                    }
                }
            }
        }
        let placement = Self {
            num_partitions: num_partitions as u32,
            node_partition,
            waves,
        };
        placement.validate(schedule)?;
        Ok(placement)
    }

    pub fn validate(&self, schedule: &HeteroSchedule) -> Result<(), PartsV2Error> {
        let np = self.num_partitions as usize;
        if np == 0 {
            return Err(PartsV2Error::ZeroPartitions);
        }
        if self.waves.len() != schedule.waves.len() {
            return Err(PartsV2Error::BadWaveCount {
                expected: schedule.waves.len(),
                got: self.waves.len(),
            });
        }
        if self.node_partition.len() != schedule.nodes.len() {
            return Err(PartsV2Error::BadNode {
                node: self.node_partition.len(),
            });
        }
        let mut seen = vec![false; schedule.nodes.len()];
        for (wave_id, partitions) in self.waves.iter().enumerate() {
            if partitions.len() != np {
                return Err(PartsV2Error::BadPartitionCount {
                    wave: wave_id,
                    expected: np,
                    got: partitions.len(),
                });
            }
            for (part, work) in partitions.iter().enumerate() {
                for node_u32 in work.all_nodes() {
                    let node = node_u32 as usize;
                    if node >= schedule.nodes.len() {
                        return Err(PartsV2Error::BadNode { node });
                    }
                    if seen[node] {
                        return Err(PartsV2Error::NodePlacedTwice { node });
                    }
                    if schedule.nodes[node].level != wave_id {
                        return Err(PartsV2Error::WrongWave {
                            node,
                            expected: schedule.nodes[node].level,
                            got: wave_id,
                        });
                    }
                    let expected = self.node_partition[node];
                    if expected != part as u32 {
                        return Err(PartsV2Error::PartitionMismatch {
                            node,
                            expected,
                            got: part as u32,
                        });
                    }
                    seen[node] = true;
                }
            }
        }
        if let Some(node) = seen.iter().position(|&value| !value) {
            return Err(PartsV2Error::MissingNode { node });
        }
        Ok(())
    }
}

impl GemPartsV2 {
    pub fn new(legacy: Vec<Vec<Partition>>, hetero: HeteroPlacementV2) -> Self {
        Self {
            magic: GEM_PARTS_V2_MAGIC,
            version: GEM_PARTS_V2_VERSION,
            legacy,
            hetero,
        }
    }

    pub fn validate(&self, schedule: &HeteroSchedule) -> Result<(), PartsV2Error> {
        if self.magic != GEM_PARTS_V2_MAGIC {
            return Err(PartsV2Error::BadMagic(self.magic));
        }
        if self.version != GEM_PARTS_V2_VERSION {
            return Err(PartsV2Error::UnsupportedVersion(self.version));
        }
        self.hetero.validate(schedule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{Carry4Block, DriverType, AIG};
    use crate::schedule::build_schedule;

    fn chained_aig() -> AIG {
        let mut aig = AIG::default();
        aig.num_aigpins = 16;
        aig.drivers = vec![DriverType::Tie0; 17];
        for pin in 1..=8 {
            aig.drivers[pin] = DriverType::InputPort(pin);
        }
        let mut a = Carry4Block::default();
        a.s_iv = [2, 4, 6, 8];
        a.di_iv = [10, 12, 14, 16];
        a.co_out = [9, 10, 11, 12];
        let mut b = Carry4Block::default();
        b.s_iv = [2, 4, 6, 8];
        b.di_iv = [10, 12, 14, 16];
        b.cin_iv = 12 << 1;
        b.co_out = [13, 14, 15, 16];
        aig.carry4s.insert(100, a);
        aig.carry4s.insert(101, b);
        for (bit, pin) in [9, 10, 11, 12].into_iter().enumerate() {
            aig.drivers[pin] = DriverType::CARRY4(100, bit + 4);
        }
        for (bit, pin) in [13, 14, 15, 16].into_iter().enumerate() {
            aig.drivers[pin] = DriverType::CARRY4(101, bit + 4);
        }
        aig
    }

    #[test]
    fn two_partition_contract_places_every_node_once_and_preserves_waves() {
        let schedule = build_schedule(&chained_aig()).unwrap();
        let placement = HeteroPlacementV2::build(&schedule, 2).unwrap();
        assert_eq!(placement.num_partitions, 2);
        assert_eq!(placement.waves.len(), 2);
        placement.validate(&schedule).unwrap();
    }

    #[test]
    fn versioned_parts_round_trip_and_reject_corruption() {
        let schedule = build_schedule(&chained_aig()).unwrap();
        let placement = HeteroPlacementV2::build(&schedule, 2).unwrap();
        let file = GemPartsV2::new(Vec::new(), placement);
        let bytes = serde_bare::to_vec(&file).unwrap();
        let mut decoded: GemPartsV2 = serde_bare::from_slice(&bytes).unwrap();
        decoded.validate(&schedule).unwrap();
        decoded.version += 1;
        assert_eq!(
            decoded.validate(&schedule),
            Err(PartsV2Error::UnsupportedVersion(GEM_PARTS_V2_VERSION + 1))
        );
    }
}
