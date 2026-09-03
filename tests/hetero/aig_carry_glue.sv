// SPDX-License-Identifier: Apache-2.0
//
// AIG glue logic feeding macro operands: `s = a & b` (4 AND gates) drives
// CARRY4.S, and `a[0] & b[0]` (1 AND gate) drives CARRY4.CI. This exercises the
// V2 evaluator's per-wave AIG fold -- the region gates run (level by level)
// before the CARRY4 of the next wave reads its inputs. The 8-bit chain result
// is C-bypassed through a DSP so it lands in persistent P.
//
//   scripts/run_v2_netlist.sh tests/hetero/aig_carry_glue.sv aig_carry_glue

module aig_carry_glue (
    input  wire        clk,
    input  wire [3:0]  a,
    input  wire [3:0]  b,
    output wire [47:0] p
);
    wire [3:0] s = a & b;          // 4 AND gates, intra-region level 0
    wire       ci = a[0] & b[0];   // 1 AND gate
    wire [3:0] o, co;

    CARRY4 c (
        .DI     (a),
        .S      (s),
        .CI     (ci),
        .CYINIT (1'b0),
        .O      (o),
        .CO     (co)
    );

    DSP48E2 #(.PREG(1), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")) sink (
        .A      (27'b0),
        .D      (27'b0),
        .B      (18'b0),
        .C      ({40'b0, co, o}),
        .OPMODE (9'h030),
        .ALUMODE(4'h0),
        .INMODE (5'b0),
        .CLK    (clk),
        .CEP    (1'b1),
        .RSTP   (1'b0),
        .P      (p)
    );
endmodule
