/*
 * Zenith Macros for Big-GEM Theory
 * These stubs are used to define the pin directions and widths for Yosys and netlistdb.
 * Yosys will see these as blackboxes during the techmap pass, preventing them from being flattened.
 */

(* blackbox *)
module DSP48E2 #(
    parameter integer PREG = 1,
    parameter USE_MULT = "MULTIPLY",
    parameter USE_SIMD = "ONE48"
)(
    input [26:0] A,
    input [26:0] D,
    input [17:0] B,
    input [47:0] C,
    input [8:0] OPMODE,
    input [3:0] ALUMODE,
    input [4:0] INMODE,
    input CLK,
    input CEP,
    input RSTP,
    output [47:0] P
);
endmodule

(* blackbox *)
module CARRY4 (
    input [3:0] DI,
    input [3:0] S,
    input CI,
    input CYINIT,
    output [3:0] O,
    output [3:0] CO
);
endmodule

(* blackbox *)
module SRLC32E #(
    parameter [31:0] INIT = 32'h00000000
)(
    input D,
    input CE,
    input CLK,
    input [4:0] A,
    output Q,
    output Q31
);
endmodule
