//! MINIMAL stub of gem::aig, field-for-field with the real one for the subset
//! that src/schedule.rs and src/macro_layout.rs touch. Used only to type/borrow
//! -check those two modules in an environment that cannot build mt-kahypar.

use indexmap::{IndexMap, IndexSet};

#[derive(Debug, Default, Clone)]
pub struct DFF {
    pub d_iv: usize,
    pub en_iv: usize,
    pub q: usize,
}

#[derive(Debug, Default, Clone)]
pub struct RAMBlock {
    pub port_r_addr_iv: [usize; 13],
    pub port_r_en_iv: usize,
    pub port_r_rd_data: [usize; 32],
    pub port_w_addr_iv: [usize; 13],
    pub port_w_wr_en_iv: [usize; 32],
    pub port_w_wr_data_iv: [usize; 32],
}

#[derive(Debug, Clone)]
pub struct DSPBlock {
    pub a_iv: [usize; 27],
    pub d_iv: [usize; 27],
    pub b_iv: [usize; 18],
    pub c_iv: [usize; 48],
    pub opmode_iv: [usize; 9],
    pub alumode_iv: [usize; 4],
    pub inmode_iv: [usize; 5],
    pub p_out: [usize; 48],
    pub clk_iv: usize,
    pub cep_iv: usize,
    pub rstp_iv: usize,
    pub preg: u32,
}

impl Default for DSPBlock {
    fn default() -> Self {
        Self {
            a_iv: [0; 27],
            d_iv: [0; 27],
            b_iv: [0; 18],
            c_iv: [0; 48],
            opmode_iv: [0; 9],
            alumode_iv: [0; 4],
            inmode_iv: [0; 5],
            p_out: [0; 48],
            clk_iv: 0,
            cep_iv: 1,
            rstp_iv: 0,
            preg: 1,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Carry4Block {
    pub di_iv: [usize; 4],
    pub s_iv: [usize; 4],
    pub cin_iv: usize,
    pub cyinit_iv: usize,
    pub o_out: [usize; 4],
    pub co_out: [usize; 4],
}

#[derive(Debug, Default, Clone)]
pub struct Srlc32eBlock {
    pub d_iv: usize,
    pub ce_iv: usize,
    pub a_iv: [usize; 5],
    pub clk_iv: usize,
    pub q_out: usize,
    pub q31_out: usize,
    pub init: u32,
}

#[derive(Debug, Clone)]
pub enum DriverType {
    AndGate(usize, usize),
    InputPort(usize),
    InputClockFlag(usize, u8),
    DFF(usize),
    SRAM(usize),
    DSP(usize, usize),
    CARRY4(usize, usize),
    SRLC32E(usize, usize),
    Tie0,
}

#[derive(Debug, Default)]
pub struct AIG {
    pub num_aigpins: usize,
    pub pin2aigpin_iv: Vec<usize>,
    pub clock_pin2aigpins: IndexMap<usize, (usize, usize)>,
    pub drivers: Vec<DriverType>,
    pub and_gate_cache: IndexMap<(usize, usize), usize>,
    pub primary_outputs: IndexSet<usize>,
    pub dffs: IndexMap<usize, DFF>,
    pub srams: IndexMap<usize, RAMBlock>,
    pub dsps: IndexMap<usize, DSPBlock>,
    pub carry4s: IndexMap<usize, Carry4Block>,
    pub srlc32es: IndexMap<usize, Srlc32eBlock>,
    pub fanouts_start: Vec<usize>,
    pub fanouts: Vec<usize>,
}
