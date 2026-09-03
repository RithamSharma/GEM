//! Independently testable, fixed-width models for the PS primitive subset.
//!
//! The CUDA implementation must match these functions bit-for-bit.  Keeping
//! the contract outside the kernel makes unsupported DSP controls explicit
//! and gives tests a CPU oracle that does not depend on GPU availability.

use std::collections::VecDeque;

const MASK_27: u64 = (1_u64 << 27) - 1;
const MASK_18: u64 = (1_u64 << 18) - 1;
const MASK_45: u64 = (1_u64 << 45) - 1;
const MASK_48: u64 = (1_u64 << 48) - 1;

fn sign_extend(value: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((value << shift) as i64) >> shift
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DspMode {
    C,
    Multiply,
    Accumulate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DspControlError {
    UnsupportedOpmode(u16),
    UnsupportedAlumode(u8),
}

/// Decode the bounded DSP48E2 subset required by the PS.
///
/// OPMODE uses the real W/X/Y/Z encoding: C=0x030, M=0x005, P+M=0x025.
/// Only ALUMODE=0 (addition) is accepted. INMODE[2] enables D in the
/// pre-adder and INMODE[3] selects subtraction, which is outside this subset.
pub fn decode_dsp_controls(
    opmode: u16,
    alumode: u8,
    inmode: u8,
) -> Result<(DspMode, bool), DspControlError> {
    if alumode & 0xf != 0 {
        return Err(DspControlError::UnsupportedAlumode(alumode & 0xf));
    }
    let mode = match opmode & 0x1ff {
        0x030 => DspMode::C,
        0x005 => DspMode::Multiply,
        0x025 => DspMode::Accumulate,
        other => return Err(DspControlError::UnsupportedOpmode(other)),
    };
    let preadd = inmode & 0b0_0100 != 0 && inmode & 0b0_1000 == 0;
    Ok((mode, preadd))
}

/// Return the exact 48-bit next P value for the supported DSP subset.
pub fn dsp48e2_next(
    a: u32,
    b: u32,
    c: u64,
    d: u32,
    p_current: u64,
    mode: DspMode,
    preadd: bool,
) -> u64 {
    let a27 = u64::from(a) & MASK_27;
    let d27 = u64::from(d) & MASK_27;
    let ad_bits = if preadd {
        a27.wrapping_add(d27) & MASK_27
    } else {
        a27
    };
    let ad = sign_extend(ad_bits, 27);
    let b18 = sign_extend(u64::from(b) & MASK_18, 18);
    let product_bits = (ad.wrapping_mul(b18) as u64) & MASK_45;
    match mode {
        DspMode::C => c & MASK_48,
        DspMode::Multiply => sign_extend(product_bits, 45) as u64 & MASK_48,
        DspMode::Accumulate => {
            sign_extend(p_current & MASK_48, 48).wrapping_add(sign_extend(product_bits, 45)) as u64
                & MASK_48
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Carry4Result {
    pub o: u8,
    pub co: u8,
}

pub fn carry4(s: u8, di: u8, ci: bool, cyinit: bool) -> Carry4Result {
    let mut carry = ci || cyinit;
    let mut o = 0_u8;
    let mut co = 0_u8;
    for bit in 0..4 {
        let select = (s >> bit) & 1 != 0;
        let data = (di >> bit) & 1 != 0;
        o |= u8::from(select ^ carry) << bit;
        carry = if select { carry } else { data };
        co |= u8::from(carry) << bit;
    }
    Carry4Result { o, co }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Srlc32eOutputs {
    pub q: bool,
    pub q31: bool,
}

pub fn srlc32e_outputs(state: u32, address: u8) -> Srlc32eOutputs {
    Srlc32eOutputs {
        q: state >> (address & 31) & 1 != 0,
        q31: state >> 31 & 1 != 0,
    }
}

pub fn srlc32e_rising_edge(state: u32, d: bool, ce: bool) -> u32 {
    if ce {
        state.wrapping_shl(1) | u32::from(d)
    } else {
        state
    }
}

/// Evaluate one simulator transaction. Storage changes on a rising edge and
/// the asynchronous taps then observe the new storage in the same timestamp.
pub fn srlc32e_step(
    state: u32,
    d: bool,
    ce: bool,
    rising_edge: bool,
    address: u8,
) -> (Srlc32eOutputs, u32) {
    let next = if rising_edge {
        srlc32e_rising_edge(state, d, ce)
    } else {
        state
    };
    (srlc32e_outputs(next, address), next)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkKind {
    Aig,
    Carry4,
    Srlc32e,
    Dsp48e2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    InvalidEdge { producer: usize, consumer: usize },
    CombinationalCycle,
}

/// Dependency levels split into type-homogeneous queues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedLevel {
    pub aig: Vec<usize>,
    pub carry4: Vec<usize>,
    pub srlc32e: Vec<usize>,
    pub dsp48e2: Vec<usize>,
}

impl TypedLevel {
    fn push(&mut self, kind: WorkKind, node: usize) {
        match kind {
            WorkKind::Aig => self.aig.push(node),
            WorkKind::Carry4 => self.carry4.push(node),
            WorkKind::Srlc32e => self.srlc32e.push(node),
            WorkKind::Dsp48e2 => self.dsp48e2.push(node),
        }
    }
}

/// Build a topological schedule. Each `(producer, consumer)` edge represents
/// same-cycle combinational visibility. Sequential current->next edges must
/// not be passed here; they cross the cycle boundary instead.
pub fn build_typed_schedule(
    kinds: &[WorkKind],
    edges: &[(usize, usize)],
) -> Result<Vec<TypedLevel>, ScheduleError> {
    let mut indegree = vec![0_usize; kinds.len()];
    let mut fanout = vec![Vec::new(); kinds.len()];
    for &(producer, consumer) in edges {
        if producer >= kinds.len() || consumer >= kinds.len() {
            return Err(ScheduleError::InvalidEdge { producer, consumer });
        }
        fanout[producer].push(consumer);
        indegree[consumer] += 1;
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(node, &degree)| (degree == 0).then_some(node))
        .collect::<VecDeque<_>>();
    let mut node_level = vec![0_usize; kinds.len()];
    let mut visited = 0;
    while let Some(node) = ready.pop_front() {
        visited += 1;
        for &consumer in &fanout[node] {
            node_level[consumer] = node_level[consumer].max(node_level[node] + 1);
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.push_back(consumer);
            }
        }
    }
    if visited != kinds.len() {
        return Err(ScheduleError::CombinationalCycle);
    }
    let mut levels = (0..node_level.iter().copied().max().unwrap_or(0) + 1)
        .map(|_| TypedLevel {
            aig: vec![],
            carry4: vec![],
            srlc32e: vec![],
            dsp48e2: vec![],
        })
        .collect::<Vec<_>>();
    for (node, &kind) in kinds.iter().enumerate() {
        levels[node_level[node]].push(kind, node);
    }
    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carry4_is_exhaustive_over_all_1024_inputs() {
        for packed in 0_u16..1024 {
            let s = (packed & 0xf) as u8;
            let di = ((packed >> 4) & 0xf) as u8;
            let ci = packed >> 8 & 1 != 0;
            let cyinit = packed >> 9 & 1 != 0;
            let got = carry4(s, di, ci, cyinit);
            let mut carry = ci || cyinit;
            let mut expected_o = 0;
            let mut expected_co = 0;
            for bit in 0..4 {
                expected_o |= u8::from(((s >> bit) & 1 != 0) ^ carry) << bit;
                carry = if (s >> bit) & 1 != 0 {
                    carry
                } else {
                    (di >> bit) & 1 != 0
                };
                expected_co |= u8::from(carry) << bit;
            }
            assert_eq!(
                got,
                Carry4Result {
                    o: expected_o,
                    co: expected_co
                }
            );
        }
    }

    #[test]
    fn dsp_uses_real_control_encodings_and_fixed_widths() {
        assert_eq!(decode_dsp_controls(0x030, 0, 0), Ok((DspMode::C, false)));
        assert_eq!(
            decode_dsp_controls(0x005, 0, 4),
            Ok((DspMode::Multiply, true))
        );
        assert_eq!(
            decode_dsp_controls(0x025, 0, 4),
            Ok((DspMode::Accumulate, true))
        );
        assert!(decode_dsp_controls(0x001, 0, 0).is_err());
        assert!(decode_dsp_controls(0x005, 3, 0).is_err());
        // Largest positive A plus one wraps to the most-negative 27-bit value.
        let p = dsp48e2_next(0x03ff_ffff, 1, 0, 1, 0, DspMode::Multiply, true);
        assert_eq!(p, ((-(1_i64 << 26)) as u64) & MASK_48);
    }

    #[test]
    fn srl_edge_updates_storage_before_asynchronous_taps_settle() {
        let initial = 0x8000_0001;
        assert_eq!(
            srlc32e_outputs(initial, 0),
            Srlc32eOutputs { q: true, q31: true }
        );
        let (outputs, next) = srlc32e_step(initial, false, true, true, 0);
        assert_eq!(next, 2);
        assert_eq!(
            outputs,
            Srlc32eOutputs {
                q: false,
                q31: false
            }
        );
        assert_eq!(srlc32e_rising_edge(next, true, false), next);
        assert_eq!(srlc32e_step(next, true, true, false, 0).1, next);
    }

    #[test]
    fn typed_schedule_preserves_mixed_macro_dependencies() {
        let kinds = [
            WorkKind::Aig,
            WorkKind::Carry4,
            WorkKind::Dsp48e2,
            WorkKind::Srlc32e,
        ];
        let schedule = build_typed_schedule(&kinds, &[(0, 1), (1, 2), (2, 3)]).unwrap();
        assert_eq!(schedule.len(), 4);
        assert_eq!(schedule[0].aig, [0]);
        assert_eq!(schedule[1].carry4, [1]);
        assert_eq!(schedule[2].dsp48e2, [2]);
        assert_eq!(schedule[3].srlc32e, [3]);
        assert_eq!(
            build_typed_schedule(&kinds[..2], &[(0, 1), (1, 0)]),
            Err(ScheduleError::CombinationalCycle)
        );
    }
}
