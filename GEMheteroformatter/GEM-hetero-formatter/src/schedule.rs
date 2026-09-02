// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Canonical heterogeneous same-cycle dependency schedule (Zenith PS, Part B/1).
//!
//! GEM's production traversal ([`AIG::topo_traverse_generic`]) only recurses
//! through [`DriverType::AndGate`]. Every macro output pin
//! (`DriverType::DSP`/`CARRY4`/`SRLC32E`) is therefore a *traversal leaf* --
//! treated exactly like a primary input or a DFF `Q`. That is correct only for
//! `DSP48E2`, whose single clocked register `PREG` makes `P` an old-state
//! value. It is *wrong* for `CARRY4` (purely combinational) and for the
//! `SRLC32E` asynchronous `Q`/`Q31` taps: a same-cycle consumer of those
//! outputs (`CARRY_A.CO[3] -> CARRY_B.CI`) currently samples the previous
//! cycle's global-state word.
//!
//! This module builds the unified graph `G = (V, E_same U E_next)` that the
//! PS requires, levelizes it, and splits every level into type-homogeneous
//! waves. It is pure host logic with no GPU dependency; the CUDA V2 interpreter
//! and the (interim) V1 staging path both consume its output.
//!
//! ## Semantics frozen for this subset
//!
//! | primitive | same-cycle outputs        | old-state outputs | next-state commit |
//! |-----------|---------------------------|-------------------|-------------------|
//! | AIG gate  | its value                 | -                 | -                 |
//! | `CARRY4`  | `O[3:0]`, `CO[3:0]`       | -                 | -                 |
//! | `SRLC32E` | `Q`, `Q31` (post-shift)   | -                 | 32-bit shift reg  |
//! | `DSP48E2` | -                         | `P[47:0]` (`PREG`)| `P_next`          |
//!
//! `E_next` (DFF `D->Q`, DSP `P_next->P`, SRLC `storage`) never contributes to
//! the same-cycle in-degree, so registered feedback is always acyclic and a
//! genuine combinational loop is the only thing [`build_schedule`] rejects.

use indexmap::{IndexMap, IndexSet};

use crate::aig::{DriverType, AIG};

/// Word-level primitive kinds handled natively by the heterogeneous engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum MacroKind {
    Carry4,
    Dsp48e2,
    Srlc32e,
}

/// A vertex of the unified graph `G`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum NodeKind {
    /// A maximal run of AIG gates that share the same *macro cut* (the set of
    /// macro outputs in their combinational fan-in). Evaluated by the existing
    /// 256-thread Boomerang fold.
    AigRegion,
    /// One preserved word-level macro instance.
    Macro(MacroKind),
}

/// One scheduled operation.
#[derive(Clone, Debug)]
pub struct HeteroNode {
    pub kind: NodeKind,
    /// `aig.carry4s` / `aig.dsps` / `aig.srlc32es` key for `Macro`; `usize::MAX`
    /// for `AigRegion`.
    pub cell_id: usize,
    /// AIG pins driven by AND gates that belong to this region (empty for
    /// macros). Sorted ascending.
    pub region_gates: Vec<usize>,
    /// AIG pins this node makes visible *in the current cycle* (region gate
    /// pins, or a macro's combinational output pins). `DSP48E2` contributes
    /// nothing here because `P` is registered.
    pub outputs: Vec<usize>,
    /// `level(v)` from the Kahn recurrence below. Filled by [`build_schedule`].
    pub level: usize,
}

impl HeteroNode {
    pub fn macro_kind(&self) -> Option<MacroKind> {
        match self.kind {
            NodeKind::Macro(k) => Some(k),
            NodeKind::AigRegion => None,
        }
    }
}

/// One dependency level, partitioned into single-`NodeKind` queues so a GPU
/// warp never runs a per-instance `switch` over primitive kinds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypedWave {
    pub level: usize,
    pub aig_regions: Vec<usize>,
    pub carry4: Vec<usize>,
    pub dsp48e2: Vec<usize>,
    pub srlc32e: Vec<usize>,
}

impl TypedWave {
    pub fn is_empty(&self) -> bool {
        self.aig_regions.is_empty()
            && self.carry4.is_empty()
            && self.dsp48e2.is_empty()
            && self.srlc32e.is_empty()
    }
}

/// Why a netlist cannot be scheduled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// A real same-cycle cycle (`E_same` is not a DAG). `nodes` names every
    /// vertex still on the cycle after Kahn peeling.
    CombinationalCycle { nodes: Vec<String> },
    /// An input pin resolves to no producer at all.
    UnresolvedDriver { aigpin: usize, consumer: String },
}

/// The finished schedule.
#[derive(Clone, Debug)]
pub struct HeteroSchedule {
    pub nodes: Vec<HeteroNode>,
    /// `(producer node, consumer node)` for every same-cycle edge, de-duplicated.
    pub edges_same: Vec<(usize, usize)>,
    /// `waves[l].level == l`; `waves.len() == num_levels`.
    pub waves: Vec<TypedWave>,
    pub num_levels: usize,
    /// AIG pin -> the node that produces it *this cycle* (AND-gate pins and
    /// `CARRY4`/`SRLC32E` output pins only; `DSP48E2` `P` pins are old-state and
    /// are deliberately absent).
    pub producer: IndexMap<usize, usize>,
}

impl HeteroSchedule {
    /// `level(cell)` for every macro instance, keyed by its `aig.*` cell id.
    /// The interim V1 path uses this to force a major-stage cut whenever a
    /// macro consumes another macro's current-cycle output.
    pub fn macro_levels(&self) -> IndexMap<usize, usize> {
        self.nodes
            .iter()
            .filter(|n| n.macro_kind().is_some())
            .map(|n| (n.cell_id, n.level))
            .collect()
    }

    /// Direct `producer_cell -> consumer_cell` macro edges that are same-cycle
    /// (`CO[3] -> CI` style). Used by tests and by the staging bridge.
    pub fn macro_to_macro_edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &(u, v) in &self.edges_same {
            if self.nodes[u].macro_kind().is_some() && self.nodes[v].macro_kind().is_some() {
                out.push((self.nodes[u].cell_id, self.nodes[v].cell_id));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// AIG pins whose value is consumed *later in the same cycle* by a
    /// higher-level node. On the interim V1 path the stager must be able to
    /// publish each of these as a `StagedAIG::primary_output_pin` so the
    /// consumer reads it from current-cycle global storage (`output_state`,
    /// the `idx >> 31` kernel path) instead of the previous-cycle
    /// `input_state`. Over-inclusive by design: `StagedAIG::from_split` keeps
    /// only the pins actually live at each chosen split level.
    pub fn cross_level_staged_pins(&self) -> IndexSet<usize> {
        let mut pins = IndexSet::new();
        for &(u, v) in &self.edges_same {
            debug_assert!(self.nodes[u].level < self.nodes[v].level);
            for &p in &self.nodes[u].outputs {
                pins.insert(p);
            }
        }
        pins
    }

    /// Distinct dependency levels at which at least one macro is scheduled,
    /// ascending. The interim path splits a major stage before each of these
    /// (after level 0) so every `macro_to_macro_edges()` pair straddles a
    /// `grid.sync()` boundary.
    pub fn macro_cut_levels(&self) -> Vec<usize> {
        let mut ls: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.macro_kind().is_some())
            .map(|n| n.level)
            .collect();
        ls.sort_unstable();
        ls.dedup();
        ls
    }
}

/// Classification of an AIG pin as a same-cycle value source.
enum PinRole {
    /// Available at cycle start (primary input, DFF `Q`, SRAM read data,
    /// `DSP48E2` `P`, constant 0). Contributes no `E_same` edge.
    OldState,
    /// Produced this cycle by `node`.
    SameCycle(usize),
}

/// Build the heterogeneous schedule for `aig`.
///
/// `level(v) = 0` when every input pin of `v` is old-state, otherwise
/// `level(v) = 1 + max { level(u) : (u, v) in E_same }`.
pub fn build_schedule(aig: &AIG) -> Result<HeteroSchedule, ScheduleError> {
    let mut nodes: Vec<HeteroNode> = Vec::new();

    // ---- 1. one node per macro instance, and the output-pin -> node map ----
    let mut macro_node_of_cell: IndexMap<usize, usize> = IndexMap::new();
    // same-cycle macro output pins only (CARRY4, SRLC32E).
    let mut producer: IndexMap<usize, usize> = IndexMap::new();

    for (&cell, blk) in &aig.carry4s {
        let idx = nodes.len();
        let outputs: Vec<usize> = blk
            .o_out
            .iter()
            .chain(blk.co_out.iter())
            .copied()
            .filter(|&p| p != 0)
            .collect();
        for &p in &outputs {
            producer.insert(p, idx);
        }
        nodes.push(HeteroNode {
            kind: NodeKind::Macro(MacroKind::Carry4),
            cell_id: cell,
            region_gates: Vec::new(),
            outputs,
            level: 0,
        });
        macro_node_of_cell.insert(cell, idx);
    }
    for (&cell, blk) in &aig.srlc32es {
        let idx = nodes.len();
        let outputs: Vec<usize> = [blk.q_out, blk.q31_out]
            .into_iter()
            .filter(|&p| p != 0)
            .collect();
        for &p in &outputs {
            producer.insert(p, idx);
        }
        nodes.push(HeteroNode {
            kind: NodeKind::Macro(MacroKind::Srlc32e),
            cell_id: cell,
            region_gates: Vec::new(),
            outputs,
            level: 0,
        });
        macro_node_of_cell.insert(cell, idx);
    }
    for (&cell, _blk) in &aig.dsps {
        // DSP `P` is registered (PREG=1): an old-state output, so it is NOT
        // registered in `producer`. The node still exists as a same-cycle
        // *sink* that consumes A/B/C/D/INMODE/OPMODE/... this cycle.
        let idx = nodes.len();
        nodes.push(HeteroNode {
            kind: NodeKind::Macro(MacroKind::Dsp48e2),
            cell_id: cell,
            region_gates: Vec::new(),
            outputs: Vec::new(),
            level: 0,
        });
        macro_node_of_cell.insert(cell, idx);
    }

    // ---- 2. macro cut of every AND-gate pin, in ascending pin id order ----
    // cut[g] = sorted unique macro node indices whose *same-cycle* output lies
    // in the AND-only fan-in of g. Monotonic along AND edges, so ascending id
    // order (which is a topological order for gates, by AIG construction) is
    // enough. Gates that share a cut form one region.
    let n = aig.num_aigpins;
    let mut cut: Vec<Vec<u32>> = vec![Vec::new(); n + 1];

    // contribution of one child pin `x` to its parent gate's cut.
    fn child_contrib(
        aig: &AIG,
        macro_node_of_cell: &IndexMap<usize, usize>,
        cut: &[Vec<u32>],
        x: usize,
    ) -> Vec<u32> {
        if x == 0 {
            return Vec::new();
        }
        match &aig.drivers[x] {
            DriverType::AndGate(..) => cut[x].clone(),
            DriverType::CARRY4(c, _) | DriverType::SRLC32E(c, _) => {
                vec![*macro_node_of_cell.get(c).expect("macro cell registered") as u32]
            }
            // DSP `P` and every genuine source contribute nothing.
            _ => Vec::new(),
        }
    }

    let mut region_of_gate: IndexMap<usize, usize> = IndexMap::new();
    let mut region_of_cut: IndexMap<Vec<u32>, usize> = IndexMap::new();
    let mut gate_ids: Vec<usize> = Vec::new();

    for g in 1..=n {
        let (a, b) = match &aig.drivers[g] {
            DriverType::AndGate(a, b) => (a >> 1, b >> 1),
            _ => continue,
        };
        let mut merged = child_contrib(aig, &macro_node_of_cell, &cut, a);
        for m in child_contrib(aig, &macro_node_of_cell, &cut, b) {
            merged.push(m);
        }
        merged.sort_unstable();
        merged.dedup();
        cut[g] = merged.clone();
        gate_ids.push(g);

        let region_idx = *region_of_cut.entry(merged).or_insert_with(|| {
            let idx = nodes.len();
            nodes.push(HeteroNode {
                kind: NodeKind::AigRegion,
                cell_id: usize::MAX,
                region_gates: Vec::new(),
                outputs: Vec::new(),
                level: 0,
            });
            idx
        });
        region_of_gate.insert(g, region_idx);
        nodes[region_idx].region_gates.push(g);
        nodes[region_idx].outputs.push(g);
        producer.insert(g, region_idx);
    }

    // ---- 3. same-cycle edges ----
    fn role(producer: &IndexMap<usize, usize>, p: usize) -> PinRole {
        if p == 0 {
            return PinRole::OldState;
        }
        match producer.get(&p) {
            Some(&node) => PinRole::SameCycle(node),
            None => PinRole::OldState,
        }
    }

    let mut edge_set: IndexSet<(usize, usize)> = IndexSet::new();
    let connect = |edge_set: &mut IndexSet<(usize, usize)>, pins: &[usize], v: usize| {
        for &p in pins {
            if let PinRole::SameCycle(u) = role(&producer, p >> 1) {
                if u != v {
                    edge_set.insert((u, v));
                }
            }
        }
    };

    // macro input pins (pin-with-invert values; `connect` shifts out the
    // invert bit). `clk_iv` is the edge selector (E_next) and is excluded.
    for (&cell, blk) in &aig.carry4s {
        let v = macro_node_of_cell[&cell];
        let mut ins: Vec<usize> = Vec::new();
        ins.extend_from_slice(&blk.di_iv);
        ins.extend_from_slice(&blk.s_iv);
        ins.push(blk.cin_iv);
        ins.push(blk.cyinit_iv);
        connect(&mut edge_set, &ins, v);
    }
    for (&cell, blk) in &aig.srlc32es {
        let v = macro_node_of_cell[&cell];
        let mut ins: Vec<usize> = vec![blk.d_iv, blk.ce_iv];
        ins.extend_from_slice(&blk.a_iv);
        connect(&mut edge_set, &ins, v);
    }
    for (&cell, blk) in &aig.dsps {
        let v = macro_node_of_cell[&cell];
        let mut ins: Vec<usize> = Vec::new();
        ins.extend_from_slice(&blk.a_iv);
        ins.extend_from_slice(&blk.d_iv);
        ins.extend_from_slice(&blk.b_iv);
        ins.extend_from_slice(&blk.c_iv);
        ins.extend_from_slice(&blk.opmode_iv);
        ins.extend_from_slice(&blk.alumode_iv);
        ins.extend_from_slice(&blk.inmode_iv);
        ins.push(blk.cep_iv);
        ins.push(blk.rstp_iv);
        connect(&mut edge_set, &ins, v);
    }

    // AIG region input pins: an AND gate fed by a macro output, or by a gate
    // that landed in a different region (its cut is a strict subset).
    for &g in &gate_ids {
        let v = region_of_gate[&g];
        let (a, b) = match &aig.drivers[g] {
            DriverType::AndGate(a, b) => (*a, *b),
            _ => unreachable!(),
        };
        connect(&mut edge_set, &[a, b], v);
    }

    let edges_same: Vec<(usize, usize)> = edge_set.iter().copied().collect();

    // ---- 4. Kahn levelization + combinational-loop rejection ----
    let mut indeg = vec![0usize; nodes.len()];
    let mut fanout: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for &(u, v) in &edges_same {
        fanout[u].push(v);
        indeg[v] += 1;
    }
    let mut ready: Vec<usize> = (0..nodes.len()).filter(|&i| indeg[i] == 0).collect();
    let mut visited = 0usize;
    let mut order: Vec<usize> = Vec::with_capacity(nodes.len());
    while let Some(u) = ready.pop() {
        visited += 1;
        order.push(u);
        for &w in &fanout[u] {
            indeg[w] -= 1;
            if indeg[w] == 0 {
                ready.push(w);
            }
        }
    }
    if visited != nodes.len() {
        let stuck: Vec<String> = indeg
            .iter()
            .enumerate()
            .filter(|(_, &d)| d != 0)
            .map(|(i, _)| node_name(&nodes[i]))
            .collect();
        return Err(ScheduleError::CombinationalCycle { nodes: stuck });
    }
    // relax in topological order (single pass suffices: `order` is a valid
    // topological order, so every predecessor is finalized first).
    for &u in &order {
        let lu = nodes[u].level;
        for &w in &fanout[u] {
            if nodes[w].level < lu + 1 {
                nodes[w].level = lu + 1;
            }
        }
    }

    let num_levels = nodes.iter().map(|nd| nd.level).max().map_or(0, |m| m + 1);
    let mut waves: Vec<TypedWave> = (0..num_levels)
        .map(|l| TypedWave {
            level: l,
            ..Default::default()
        })
        .collect();
    for (i, nd) in nodes.iter().enumerate() {
        let w = &mut waves[nd.level];
        match nd.kind {
            NodeKind::AigRegion => w.aig_regions.push(i),
            NodeKind::Macro(MacroKind::Carry4) => w.carry4.push(i),
            NodeKind::Macro(MacroKind::Dsp48e2) => w.dsp48e2.push(i),
            NodeKind::Macro(MacroKind::Srlc32e) => w.srlc32e.push(i),
        }
    }

    Ok(HeteroSchedule {
        nodes,
        edges_same,
        waves,
        num_levels,
        producer,
    })
}

fn node_name(nd: &HeteroNode) -> String {
    match nd.kind {
        NodeKind::AigRegion => format!("aig_region(cut_gates={})", nd.region_gates.len()),
        NodeKind::Macro(MacroKind::Carry4) => format!("CARRY4#cell{}", nd.cell_id),
        NodeKind::Macro(MacroKind::Dsp48e2) => format!("DSP48E2#cell{}", nd.cell_id),
        NodeKind::Macro(MacroKind::Srlc32e) => format!("SRLC32E#cell{}", nd.cell_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{Carry4Block, DriverType, Srlc32eBlock, AIG};
    use indexmap::IndexMap;

    /// Minimal hand-built AIG. `drivers[0]` is `Tie0`, matching `AIG::new`.
    struct Builder {
        aig: AIG,
    }
    impl Builder {
        fn new() -> Self {
            let mut aig = AIG::default();
            aig.num_aigpins = 0;
            aig.drivers = vec![DriverType::Tie0];
            Builder { aig }
        }
        fn input(&mut self) -> usize {
            self.aig.num_aigpins += 1;
            self.aig.drivers.push(DriverType::InputPort(0));
            self.aig.num_aigpins
        }
        fn dff_q(&mut self) -> usize {
            self.aig.num_aigpins += 1;
            self.aig.drivers.push(DriverType::DFF(self.aig.dffs.len()));
            self.aig.num_aigpins
        }
        fn and(&mut self, a_iv: usize, b_iv: usize) -> usize {
            self.aig.num_aigpins += 1;
            self.aig.drivers.push(DriverType::AndGate(a_iv, b_iv));
            self.aig.num_aigpins
        }
        /// Register a CARRY4 with the given input pin-with-invert values; the
        /// 8 output pins are freshly allocated and returned as `(O[0..4], CO[0..4])`.
        fn carry4(&mut self, cell: usize, di: [usize; 4], s: [usize; 4], cin_iv: usize, cyinit_iv: usize) -> ([usize; 4], [usize; 4]) {
            let mut o_out = [0usize; 4];
            let mut co_out = [0usize; 4];
            for k in 0..4 {
                self.aig.num_aigpins += 1;
                self.aig.drivers.push(DriverType::CARRY4(cell, k));
                o_out[k] = self.aig.num_aigpins;
            }
            for k in 0..4 {
                self.aig.num_aigpins += 1;
                self.aig.drivers.push(DriverType::CARRY4(cell, k + 4));
                co_out[k] = self.aig.num_aigpins;
            }
            let blk = Carry4Block {
                di_iv: di,
                s_iv: s,
                cin_iv,
                cyinit_iv,
                o_out,
                co_out,
            };
            self.aig.carry4s.insert(cell, blk);
            (o_out, co_out)
        }
        fn srlc32e(&mut self, cell: usize, d_iv: usize, ce_iv: usize, a_iv: [usize; 5], clk_iv: usize) -> (usize, usize) {
            self.aig.num_aigpins += 1;
            self.aig.drivers.push(DriverType::SRLC32E(cell, 0));
            let q = self.aig.num_aigpins;
            self.aig.num_aigpins += 1;
            self.aig.drivers.push(DriverType::SRLC32E(cell, 1));
            let q31 = self.aig.num_aigpins;
            self.aig.srlc32es.insert(
                cell,
                Srlc32eBlock {
                    d_iv,
                    ce_iv,
                    a_iv,
                    clk_iv,
                    q_out: q,
                    q31_out: q31,
                    init: 0,
                },
            );
            (q, q31)
        }
        fn finish(mut self) -> AIG {
            // schedule.rs does not use the fanout CSR, but keep the struct
            // internally consistent for any other consumer in a test.
            self.aig.fanouts_start = vec![0; self.aig.num_aigpins + 2];
            self.aig.fanouts = vec![];
            self.aig
        }
    }

    fn iv(pin: usize, inv: usize) -> usize {
        pin << 1 | inv
    }

    #[test]
    fn carry_chain_is_one_direct_same_cycle_edge() {
        // CARRY_A.CO[3] -> CARRY_B.CI, no AIG node between them.
        let mut b = Builder::new();
        let x = b.input();
        let (_o_a, co_a) = b.carry4(10, [iv(x, 0); 4], [iv(x, 0); 4], iv(x, 0), 0);
        let (_o_b, _co_b) = b.carry4(11, [iv(x, 0); 4], [iv(x, 0); 4], iv(co_a[3], 0), 0);
        let sched = build_schedule(&b.finish()).unwrap();

        let m2m = sched.macro_to_macro_edges();
        assert_eq!(m2m, vec![(10, 11)], "exactly one direct CARRY4->CARRY4 edge");

        let lv = sched.macro_levels();
        assert_eq!(lv[&10], 0);
        assert_eq!(lv[&11], 1, "consumer CARRY4 sits one wave later");
        assert_eq!(sched.num_levels, 2);
        assert_eq!(sched.waves[0].carry4.len(), 1);
        assert_eq!(sched.waves[1].carry4.len(), 1);
    }

    #[test]
    fn aig_carry_aig_is_ordered() {
        // in -> AND g1 -> CARRY.S -> CARRY.O[0] -> AND g2
        let mut b = Builder::new();
        let x = b.input();
        let y = b.input();
        let g1 = b.and(iv(x, 0), iv(y, 0));
        let (o, _co) = b.carry4(1, [iv(g1, 0); 4], [iv(g1, 0); 4], 0, 0);
        let g2 = b.and(iv(o[0], 0), iv(x, 1));
        let sched = build_schedule(&b.finish()).unwrap();

        let carry_node = sched.nodes.iter().position(|nd| nd.cell_id == 1).unwrap();
        let r1 = sched.producer[&g1];
        let r2 = sched.producer[&g2];
        assert!(sched.nodes[r1].level < sched.nodes[carry_node].level);
        assert!(sched.nodes[carry_node].level < sched.nodes[r2].level);
        // every serialized same-cycle edge respects strict level order.
        for &(u, v) in &sched.edges_same {
            assert!(sched.nodes[u].level < sched.nodes[v].level);
        }
    }

    #[test]
    fn one_macro_output_fans_out_to_aig_and_two_macros() {
        let mut b = Builder::new();
        let x = b.input();
        let (o, co) = b.carry4(1, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        // fanout 1: AIG
        let _g = b.and(iv(o[0], 0), iv(x, 0));
        // fanout 2 + 3: two more CARRY4s consuming co[3]
        b.carry4(2, [iv(co[3], 0); 4], [iv(x, 0); 4], 0, 0);
        b.carry4(3, [iv(x, 0); 4], [iv(x, 0); 4], iv(co[3], 0), 0);
        let sched = build_schedule(&b.finish()).unwrap();
        let mut m2m = sched.macro_to_macro_edges();
        m2m.sort();
        assert_eq!(m2m, vec![(1, 2), (1, 3)]);
        assert_eq!(sched.macro_levels()[&1], 0);
        assert_eq!(sched.macro_levels()[&2], 1);
        assert_eq!(sched.macro_levels()[&3], 1);
    }

    #[test]
    fn registered_feedback_is_accepted_as_next_cycle() {
        // SRLC.Q -> AND -> (would be DFF.D); DFF.Q -> SRLC.D. The DFF Q is a
        // source, so no same-cycle cycle exists.
        let mut b = Builder::new();
        let qd = b.dff_q();
        let ce = b.input();
        let (q, _q31) = b.srlc32e(1, iv(qd, 0), iv(ce, 0), [0; 5], 0);
        let _g = b.and(iv(q, 0), iv(ce, 0));
        let sched = build_schedule(&b.finish()).expect("registered feedback must schedule");
        assert_eq!(sched.macro_levels()[&1], 0);
    }

    #[test]
    fn genuine_combinational_loop_is_rejected() {
        // CARRY_A.CO[3] -> CARRY_B.CI and CARRY_B.CO[3] -> CARRY_A.CI.
        let mut b = Builder::new();
        let x = b.input();
        // allocate B's outputs first so we can wire A->B and B->A.
        let (_o_b, co_b) = b.carry4(2, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        let (_o_a, co_a) = b.carry4(1, [iv(x, 0); 4], [iv(x, 0); 4], iv(co_b[3], 0), 0);
        // rewrite B.cin to depend on A.
        b.aig.carry4s.get_mut(&2).unwrap().cin_iv = iv(co_a[3], 0);
        let err = build_schedule(&b.finish()).unwrap_err();
        match err {
            ScheduleError::CombinationalCycle { nodes } => {
                assert!(nodes.iter().any(|s| s.contains("CARRY4")));
            }
            other => panic!("expected CombinationalCycle, got {other:?}"),
        }
    }

    #[test]
    fn every_operation_is_scheduled_exactly_once() {
        let mut b = Builder::new();
        let x = b.input();
        let g1 = b.and(iv(x, 0), iv(x, 1));
        let (o, _co) = b.carry4(1, [iv(g1, 0); 4], [iv(g1, 0); 4], 0, 0);
        let _g2 = b.and(iv(o[0], 0), iv(o[1], 0));
        b.srlc32e(2, iv(x, 0), iv(x, 0), [0; 5], 0);
        let sched = build_schedule(&b.finish()).unwrap();

        let mut seen: IndexMap<usize, u32> = IndexMap::new();
        for w in &sched.waves {
            for &i in w
                .aig_regions
                .iter()
                .chain(&w.carry4)
                .chain(&w.dsp48e2)
                .chain(&w.srlc32e)
            {
                *seen.entry(i).or_default() += 1;
            }
        }
        assert_eq!(seen.len(), sched.nodes.len());
        assert!(seen.values().all(|&c| c == 1));
    }

    #[test]
    fn two_chained_carry4s_occupy_different_waves() {
        let mut b = Builder::new();
        let x = b.input();
        let (_o, co) = b.carry4(1, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        b.carry4(2, [iv(x, 0); 4], [iv(x, 0); 4], iv(co[3], 0), 0);
        let sched = build_schedule(&b.finish()).unwrap();
        assert_ne!(sched.macro_levels()[&1], sched.macro_levels()[&2]);
        assert!(sched.waves[0].carry4.contains(
            &sched.nodes.iter().position(|nd| nd.cell_id == 1).unwrap()
        ));
    }

    #[test]
    fn independent_same_type_macros_share_one_queue() {
        let mut b = Builder::new();
        let x = b.input();
        b.carry4(1, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        b.carry4(2, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        b.carry4(3, [iv(x, 0); 4], [iv(x, 0); 4], 0, 0);
        let sched = build_schedule(&b.finish()).unwrap();
        assert_eq!(sched.num_levels, 1);
        assert_eq!(sched.waves[0].carry4.len(), 3);
    }
}
