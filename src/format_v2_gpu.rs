// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! GPU-facing assembly of the V2 macro program (Host-Side Macro Memory Formatter
//! plan, Phase 5, Rust half).
//!
//! Takes the pure-host [`ResolvedProgram`] and produces the exact buffers a
//! single H2D upload transfers:
//!
//! * `program` -- **one** immutable `UVec<u64>` (header + every section). One
//!   base pointer, one `AsUPtr::as_uptr` upload, section offsets read from the
//!   header. No per-section FFI pointers.
//! * `gather_rounds` / `staged_word_slots` -- the Phase-4 preamble plan, so the
//!   kernel knows which old-state `u32` words to coalesce-load into which
//!   shared `u64` slots before the macro waves.
//! * `initial_state` -- the persistent `u32` image (all zero: PS
//!   "all internal macro registers initialize to zero").
//!
//! The launch wrappers are generated from `csrc/kernel_v2.cu`. They are used by
//! both `formatter_gpu_test` and the normal `cuda_test --v2` simulator path.

use ulib::UVec;

use crate::format_v2::ProgramLayoutV2;
use crate::format_v2_build::ResolvedProgram;

pub struct FlattenedScriptV2 {
    /// header + all selector sections, one immutable buffer.
    pub program: UVec<u64>,
    pub layout: ProgramLayoutV2,
    /// persistent `u32` image after each cycle is written back here; starts zero.
    pub initial_state: Vec<u32>,
    pub persistent_words: u32,
    pub current_stage_words: u32,
    pub num_partitions: u32,
    pub sram_storage_words: u32,
    pub shared_words_per_block: u32,
    /// FNV-1a the host computed over the section bytes; the GPU self-check
    /// recomputes it and compares (plan Phase 5 / 6).
    pub content_hash: u64,
    /// flat coalesced preamble plan: `[n_rounds, (len, w0..w_{len-1})...]`.
    pub gather_rounds: Vec<u32>,
    /// flat `(old_state_u32_word, shared_u64_word)` pairs the preamble stages.
    pub staged_word_slots: Vec<u32>,
}

impl FlattenedScriptV2 {
    pub fn from_resolved(rp: ResolvedProgram) -> Self {
        let content_hash = rp.layout.header.content_hash;
        let persistent_words = rp.state.persistent_words;
        let current_stage_words = rp.current_stage_words;
        let num_partitions = rp.num_partitions;
        let sram_storage_words = rp.sram_storage_words;
        let shared_words_per_block = rp.layout.header.shared_words_per_block;

        let mut gather_rounds = vec![rp.gather.rounds.len() as u32];
        for r in &rp.gather.rounds {
            gather_rounds.push(r.words.len() as u32);
            gather_rounds.extend_from_slice(&r.words);
        }
        let mut staged_word_slots = Vec::with_capacity(rp.gather.staged_word_slot.len() * 2);
        for (&orig_word, &shared_word) in &rp.gather.staged_word_slot {
            staged_word_slots.push(orig_word);
            staged_word_slots.push(shared_word);
        }

        let ResolvedProgram {
            program,
            layout,
            initial_state,
            ..
        } = rp;
        FlattenedScriptV2 {
            program: program.into(),
            layout,
            initial_state,
            persistent_words,
            current_stage_words,
            num_partitions,
            sram_storage_words,
            shared_words_per_block,
            content_hash,
            gather_rounds,
            staged_word_slots,
        }
    }

    /// Bytes a single H2D upload transfers for the immutable program. Nsight
    /// Systems (plan Phase 7) must show exactly one transfer of this size and
    /// none inside the cycle loop.
    pub fn program_bytes(&self) -> usize {
        self.program.len() * core::mem::size_of::<u64>()
    }

    /// `true` iff the assembled program passes the host validator -- the same
    /// check the GPU loader runs before dereferencing any section.
    pub fn validates(&self) -> bool {
        crate::format_v2::validate(&self.layout, self.program.len() as u64).is_ok()
    }
}
