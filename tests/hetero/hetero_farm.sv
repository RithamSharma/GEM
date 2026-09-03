// SPDX-License-Identifier: Apache-2.0
//
// Part D throughput fixture: a WIDE, SHALLOW farm of N independent
// heterogeneous units. Each unit is one copy of the exact control config the
// 300-cycle differential already verifies:
//
//     DSP48E2  PREG=1, OPMODE 9'h025 (P += A*B), INMODE 5'b00100, ALUMODE 0
//     CARRY4   standard CI / CYINIT / DI / S -> O / CO
//     SRLC32E  INIT forced to 0 (PS Zenith init constraint)
//
// The units are mutually independent -- no macro feeds another -- so the V2
// schedule is a SINGLE dependency wave of 3*N macro instances. That is the
// workload shape where evaluating macros natively on the GPU ALU should beat
// shredding them to a large AIG: one grid barrier per cycle instead of one per
// carry-chain stage.
//
// Every macro output is a primary output, so none is removed as dead code.
// All sequential state lives in the DSP PREG and the SRLC shift register.
//
// Shredded (scripts/synth_baseline.ys) each DSP48E2 lowers to ~1.5k AIG gates,
// so N=32 is ~55k gates + ~2k flip-flops vs ~96 preserved macro nodes.
//
//   ./scripts/run_partd_benchmark.sh tests/hetero/hetero_farm.sv hetero_farm 2000

module hetero_farm #(
    parameter integer N = 32
) (
    input  wire            clk,
    input  wire [26:0]     a,
    input  wire [26:0]     d,
    input  wire [17:0]     b,
    input  wire [47:0]     c,
    input  wire [3:0]      s,
    input  wire [3:0]      di,
    input  wire            ci,
    input  wire            cyinit,
    input  wire            srl_d,
    input  wire            srl_ce,
    input  wire [4:0]      srl_a,
    output wire [48*N-1:0] p_all,     // per-lane DSP accumulator
    output wire [8*N-1:0]  oco_all,   // per-lane {CO[3:0], O[3:0]}
    output wire [2*N-1:0]  q_all      // per-lane {Q31, Q}
);
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : unit
            wire [26:0] ai = a ^ i;   // distinct A per lane

            DSP48E2 #(.PREG(1), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")) dsp (
                .A      (ai),
                .D      (d),
                .B      (b),
                .C      (c),
                .OPMODE (9'h025),
                .ALUMODE(4'h0),
                .INMODE (5'b00100),
                .CLK    (clk),
                .CEP    (1'b1),
                .RSTP   (1'b0),
                .P      (p_all[48*i +: 48])
            );

            CARRY4 carry (
                .DI     (di),
                .S      (s),
                .CI     (ci),
                .CYINIT (cyinit),
                .O      (oco_all[8*i     +: 4]),
                .CO     (oco_all[8*i + 4 +: 4])
            );

            SRLC32E #(.INIT(32'h0000_0000)) srl (
                .D  (srl_d),
                .CE (srl_ce),
                .CLK(clk),
                .A  (srl_a),
                .Q  (q_all[2*i]),
                .Q31(q_all[2*i + 1])
            );
        end
    endgenerate
endmodule
