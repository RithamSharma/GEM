// Independent behavioral oracle used only by the 300-cycle regression.
// These definitions deliberately do not include aigpdk/zenith_macros.v.

module DSP48E2 #(
    parameter integer PREG = 1,
    parameter USE_MULT = "MULTIPLY",
    parameter USE_SIMD = "ONE48"
)(
    input  wire [26:0] A,
    input  wire [26:0] D,
    input  wire [17:0] B,
    input  wire [47:0] C,
    input  wire  [8:0] OPMODE,
    input  wire  [3:0] ALUMODE,
    input  wire  [4:0] INMODE,
    input  wire        CLK,
    input  wire        CEP,
    input  wire        RSTP,
    output reg  [47:0] P
);
    reg signed [26:0] ad;
    reg signed [17:0] b_signed;
    reg signed [44:0] product;
    reg signed [47:0] next_p;

    initial P = 48'b0;

    always @* begin
        ad = $signed(A);
        if (INMODE[2] && !INMODE[3])
            ad = $signed(A + D);
        b_signed = $signed(B);
        product = ad * b_signed;
        case (OPMODE)
            9'h030: next_p = $signed(C);
            9'h005: next_p = {{3{product[44]}}, product};
            9'h025: next_p = $signed(P) + {{3{product[44]}}, product};
            default: next_p = 48'bx;
        endcase
        if (ALUMODE != 4'b0000)
            next_p = 48'bx;
    end

    always @(posedge CLK) begin
        if (RSTP)
            P <= 48'b0;
        else if (CEP)
            P <= next_p;
    end
endmodule

module CARRY4 (
    input  wire [3:0] DI,
    input  wire [3:0] S,
    input  wire       CI,
    input  wire       CYINIT,
    output reg  [3:0] O,
    output reg  [3:0] CO
);
    integer bit_index;
    reg carry;
    always @* begin
        carry = CI | CYINIT;
        O = 4'b0;
        CO = 4'b0;
        for (bit_index = 0; bit_index < 4; bit_index = bit_index + 1) begin
            O[bit_index] = S[bit_index] ^ carry;
            carry = S[bit_index] ? carry : DI[bit_index];
            CO[bit_index] = carry;
        end
    end
endmodule

module SRLC32E #(
    parameter [31:0] INIT = 32'h00000000
)(
    input  wire       D,
    input  wire       CE,
    input  wire       CLK,
    input  wire [4:0] A,
    output wire       Q,
    output wire       Q31
);
    reg [31:0] state;
    initial state = INIT;
    assign Q = state[A];
    assign Q31 = state[31];
    always @(posedge CLK)
        if (CE)
            state <= {state[30:0], D};
endmodule
