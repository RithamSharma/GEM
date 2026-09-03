module carry_chain2 (
    input  wire       clk,
    input  wire [7:0] di,
    input  wire [7:0] s,
    input  wire       ci,
    output wire [7:0] o,
    output wire [7:0] co,
    output reg        sampled_carry
);
    CARRY4 low (
        .DI(di[3:0]), .S(s[3:0]), .CI(ci), .CYINIT(1'b0),
        .O(o[3:0]), .CO(co[3:0])
    );
    CARRY4 high (
        .DI(di[7:4]), .S(s[7:4]), .CI(co[3]), .CYINIT(1'b0),
        .O(o[7:4]), .CO(co[7:4])
    );

    // Keeps a real clock in the synthesized design and checks that a direct
    // macro result can also cross the cycle boundary through an ordinary DFF.
    always @(posedge clk)
        sampled_carry <= co[7];
endmodule
