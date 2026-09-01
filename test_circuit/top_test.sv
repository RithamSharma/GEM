`timescale 1ns / 1ps

module top_test (
    input  wire        clk,
    input  wire [26:0] A,
    input  wire [17:0] B,
    input  wire [47:0] C,
    input  wire [26:0] D,
    input  wire [3:0]  S,
    input  wire [3:0]  DI,
    input  wire        CYINIT,
    output wire [47:0] P_out,
    output wire [3:0]  O_out,
    output wire [3:0]  CO_out
);

    // 1. DSP48E2 Instance (Multiply-Accumulate Mode: OPMODE = 2'b10)
    DSP48E2 #(
        .AREG(0), .BREG(0), .CREG(0), .DREG(0), .ADREG(0), .MREG(0), .PREG(1),
        .USE_MULT("MULTIPLY")
    ) dsp_inst (
        .CLK(clk),
        .A(A),
        .B(B),
        .C(C),
        .D(D),
        .P(P_out),
        .OPMODE(9'b000100101) // Simplified accumulator mode
    );

    // 2. CARRY4 Instance
    CARRY4 carry_inst (
        .CI(1'b0),
        .CYINIT(CYINIT),
        .DI(DI),
        .S(S),
        .O(O_out),
        .CO(CO_out)
    );

endmodule
// --- Blackbox Definitions for Yosys ---
(* blackbox *)
module DSP48E2 #(
    parameter AREG=1, BREG=1, CREG=1, DREG=1, ADREG=1, MREG=1, PREG=1,
    parameter USE_MULT="MULTIPLY"
) (
    input wire CLK,
    input wire [26:0] A, D,
    input wire [17:0] B,
    input wire [47:0] C,
    input wire [8:0] OPMODE,
    output wire [47:0] P
);
endmodule

(* blackbox *)
module CARRY4 (
    input wire CI, CYINIT,
    input wire [3:0] DI, S,
    output wire [3:0] O, CO
);
endmodule

(* blackbox *)
module SRLC32E (
    input wire CLK, CE, D,
    input wire [4:0] A,
    output wire Q, Q31
);
endmodule
