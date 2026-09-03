// SPDX-License-Identifier: Apache-2.0
//
// Part D throughput benchmark: macro-dense by design.
//   * 16 independent DSP48E2 multiply-accumulate lanes (P += A*B).  Each one
//     lowers to ~1.5k AIG gates when the macro is NOT preserved, so the design
//     is ~24k gates in the baseline flow vs 16 macro nodes preserved.
//   * an 8-stage CARRY4 adder with the classic CO[3] -> CI chain, its S inputs
//     driven by XOR glue logic (exercises the scheduler + the AIG fold).
//
// All sequential state lives inside the DSP PREG registers, so there are no
// plain flip-flops outside the macros -- same shape as preservation_top.sv,
// which is what synth_zenith.ys's flow expects. Single clock domain.
//
// Synthesizes both ways: preserved (synth_zenith.ys) and shredded
// (scripts/synth_baseline.ys).

module bench_mac (
    input  wire        clk,
    input  wire [26:0] xin,
    input  wire [17:0] win,
    output wire [47:0] acc0,     // MAC lane 0 accumulator (persistent P)
    output wire [31:0] sum       // combinational p0[31:0] + p1[31:0]
);
    localparam integer N = 16;

    wire [47:0] p [0:N-1];
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : mac
            DSP48E2 #(.PREG(1), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")) d (
                .A      (xin),
                .D      (27'b0),
                .B      (win),
                .C      (48'b0),
                .OPMODE (9'h025),      // P_next = P + A*B  (same as preservation_top.sv)
                .ALUMODE(4'h0),
                .INMODE (5'b00100),    // pre-adder path; D=0 so AD = A  (matches the verified fixture)
                .CLK    (clk),
                .CEP    (1'b1),
                .RSTP   (1'b0),
                .P      (p[i])
            );
        end
    endgenerate
    assign acc0 = p[0];

    // 32-bit adder: p[0][31:0] + p[1][31:0] via 8 CARRY4s, carry[i] chained.
    wire [31:0] av = p[0][31:0];
    wire [31:0] bv = p[1][31:0];
    wire [31:0] sv = av ^ bv;         // XOR glue -> CARRY4.S
    wire [8:0]  carry;               // carry[0] tied low; carry[i+1] = stage i CO[3]
    wire [3:0]  ob  [0:7];
    wire [3:0]  cob [0:7];
    assign carry[0] = 1'b0;
    generate
        for (i = 0; i < 8; i = i + 1) begin : add
            CARRY4 c (
                .DI     (av[4*i +: 4]),
                .S      (sv[4*i +: 4]),
                .CI     (carry[i]),
                .CYINIT (1'b0),
                .O      (ob[i]),
                .CO     (cob[i])
            );
            assign carry[i + 1] = cob[i][3];
        end
    endgenerate
    assign sum = {ob[7], ob[6], ob[5], ob[4], ob[3], ob[2], ob[1], ob[0]};
endmodule
