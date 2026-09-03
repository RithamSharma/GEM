module sram_top (
    input  wire        clk,
    input  wire        we,
    input  wire [12:0] raddr,
    input  wire [12:0] waddr,
    input  wire [31:0] wdata,
    output reg  [31:0] q
);
    reg [31:0] mem [0:8191];
    always @(posedge clk) begin
        q <= mem[raddr];
        if (we)
            mem[waddr] <= wdata;
    end
endmodule
