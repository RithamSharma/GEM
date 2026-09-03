`timescale 1ns / 1ps

module tb_top;
    reg clk = 0;
    reg [26:0] A;
    reg [17:0] B;
    reg [47:0] C;
    reg [26:0] D;
    reg [3:0]  S;
    reg [3:0]  DI;
    reg        CYINIT;

    wire [47:0] P_out;
    wire [3:0]  O_out;
    wire [3:0]  CO_out;

    top_test uut (
        .clk(clk), .A(A), .B(B), .C(C), .D(D),
        .S(S), .DI(DI), .CYINIT(CYINIT),
        .P_out(P_out), .O_out(O_out), .CO_out(CO_out)
    );

    always #5 clk = ~clk; // 10ns clock period

    initial begin
        // Dump waveform for GEM comparison
        $dumpfile("golden_output.vcd");
        $dumpvars(0, tb_top);

        // Initialize values
        A = 27'd10; B = 18'd5; C = 48'd0; D = 27'd0;
        S = 4'b0101; DI = 4'b0011; CYINIT = 1'b0;

        #10;
        A = 27'd20; B = 18'd3; S = 4'b1111; DI = 4'b0000; CYINIT = 1'b1;
        #10;
        A = -27'sd15; B = 18'd4; S = 4'b1010; DI = 4'b1100;
        #20;

        $finish;
    end
endmodule
