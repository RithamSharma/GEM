// SPDX-License-Identifier: Apache-2.0
//
// An 8-deep CARRY4 cascade: stage k's carry-in is stage k-1's CO[3], a direct
// macro-to-macro same-cycle net with no boolean node between them (PS Part B).
// The 8-bit chain result is routed through a C-bypass DSP48E2 so it lands in a
// persistent (clocked) P register and is observable in the differential test.
//
//   scripts/run_carry_chain8_v2.sh          # synth + GPU-vs-CPU differential
//
// Expected schedule: 8 CARRY4 waves (0..7), DSP at wave 8; 9 waves total;
// 7 direct CO[3] -> CIN dependencies plus 8 O/CO -> C[..] into the DSP.

module carry_chain8 (
    input  wire        clk,
    input  wire [3:0]  x,
    output wire [47:0] p
);
    wire [3:0] o  [0:7];
    wire [3:0] co [0:7];

    genvar k;
    generate
        for (k = 0; k < 8; k = k + 1) begin : stage
            CARRY4 c (
                .DI     (x),
                .S      (x),
                .CI     (k == 0 ? 1'b0 : co[k-1][3]),
                .CYINIT (1'b0),
                .O      (o[k]),
                .CO     (co[k])
            );
        end
    endgenerate

    // OPMODE 9'h030 = C-bypass: P_next = C. The low 8 bits of C carry the
    // final CARRY4's O/CO, so the whole cascade result is committed to P.
    DSP48E2 #(.PREG(1), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")) sink (
        .A      (27'b0),
        .D      (27'b0),
        .B      (18'b0),
        .C      ({40'b0, co[7], o[7]}),
        .OPMODE (9'h030),
        .ALUMODE(4'h0),
        .INMODE (5'b0),
        .CLK    (clk),
        .CEP    (1'b1),
        .RSTP   (1'b0),
        .P      (p)
    );
endmodule
