module preservation_top (
    input  wire        clk,
    input  wire [26:0] a,
    input  wire [26:0] d,
    input  wire [17:0] b,
    input  wire [47:0] c,
    input  wire  [3:0] s,
    input  wire  [3:0] di,
    input  wire        ci,
    input  wire        cyinit,
    input  wire        srl_d,
    input  wire        srl_ce,
    input  wire  [4:0] srl_a,
    output wire [47:0] p,
    output wire  [3:0] o,
    output wire  [3:0] co,
    output wire        q,
    output wire        q31
);
    DSP48E2 #(.PREG(1), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")) dsp (
        .A(a), .D(d), .B(b), .C(c), .OPMODE(9'h025),
        .ALUMODE(4'h0), .INMODE(5'b00100), .CLK(clk),
        .CEP(1'b1), .RSTP(1'b0), .P(p)
    );
    CARRY4 carry (
        .DI(di), .S(s), .CI(ci), .CYINIT(cyinit), .O(o), .CO(co)
    );
    SRLC32E #(.INIT(32'h0000_0000)) srl (
        .D(srl_d), .CE(srl_ce), .CLK(clk), .A(srl_a), .Q(q), .Q31(q31)
    );
endmodule
