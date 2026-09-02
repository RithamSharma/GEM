`timescale 1ns / 1ps

module top_test_behavioral (
    input  wire        clk,
    input  wire [26:0] A,
    input  wire [17:0] B,
    input  wire [47:0] C,
    input  wire [26:0] D,
    input  wire [3:0]  S,
    input  wire [3:0]  DI,
    input  wire        CYINIT,
    output reg  [47:0] P_out,
    output wire [3:0]  O_out,
    output wire [3:0]  CO_out
);

    // 1. Behavioral equivalent of DSP48E2 Multiply-Accumulate
    // P_out = A * B + C (synchronous to CLK)
    always @(posedge clk) begin
        P_out <= (A * B) + C;
    end

    // 2. Behavioral equivalent of CARRY4
    // O = S ^ DI, CO = carry logic
    assign {CO_out, O_out} = DI + S + CYINIT;

endmodule
