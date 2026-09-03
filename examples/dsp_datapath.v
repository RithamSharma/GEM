// ===========================================================================
//  Example design for the GEM heterogeneous-macro flow
//  Takneek PS Zenith  --  "The Big-GEM Theory"
//
//  A small signal-processing datapath that exercises ALL THREE preserved macros
//  and contains same-cycle macro->consumer edges that FORCE the V2 wave engine
//  (the classic batched V1 path would read stale state on those edges):
//
//    * SRLC32E  - a 32-cycle shift-register DELAY LINE on sample[0]; the taps
//                 Q (position 5) and Q31 are combined into `prbs`. No feedback
//                 into D -- an SRLC32E with D = f(Q) is a combinational loop the
//                 scheduler cannot prove acyclic and will reject.
//    * DSP48E2  - tap 0: sample * 21          (OPMODE 9'h005, multiply)
//                 tap 1: accumulator P += sample * 13   (OPMODE 9'h025)
//    * CARRY4   - an 8-bit ripple adder from TWO chained CARRY4s:
//                 c_lo.CO[3] -> c_hi.CI   <-- direct macro->macro, same cycle
//    * AIG glue - XORs on the adder select lines; `mac = p0 + p1` adds the two
//                 DSP outputs in the same cycle (another macro->AIG edge)
//
//  Ports (match these names in the stimulus CSV header):
//    clk, rst, sample[15:0], addend[7:0]  ->  inputs
//    mac[47:0], sum[7:0], sum_carry, prbs ->  outputs
// ===========================================================================
module judge_dsp_datapath (
    input         clk,
    input         rst,
    input  [15:0] sample,
    input  [7:0]  addend,
    output [47:0] mac,
    output [7:0]  sum,
    output        sum_carry,
    output        prbs
);

    // ---------------------------------------------------------------------
    //  SRLC32E : 32-cycle delay line on sample[0]
    //  Q  = the bit shifted in 6 cycles ago (tap A = 5)
    //  Q31 = the bit shifted in 32 cycles ago
    // ---------------------------------------------------------------------
    wire q, q31;

    SRLC32E #(.INIT(32'h1234ABCD)) srl (
        .D  (sample[0] & ~rst),
        .CE (1'b1),
        .CLK(clk),
        .A  (5'd5),
        .Q  (q),
        .Q31(q31)
    );
    assign prbs = q ^ q31;

    // ---------------------------------------------------------------------
    //  DSP48E2 tap 0 : P = sample * 21          (OPMODE 9'h005 = multiply)
    // ---------------------------------------------------------------------
    wire [47:0] p0;
    DSP48E2 #(.PREG(1)) dsp0 (
        .A     (27'd21),
        .B     ({2'b00, sample}),
        .C     (48'd0),
        .D     (27'd0),
        .OPMODE(9'h005),
        .ALUMODE(4'h0),
        .INMODE(5'h00),
        .CLK   (clk),
        .CEP   (~rst),
        .RSTP  (rst),
        .P     (p0)
    );

    // ---------------------------------------------------------------------
    //  DSP48E2 tap 1 : running accumulator  P <= P + sample * 13
    //                                           (OPMODE 9'h025 = P + A*B)
    // ---------------------------------------------------------------------
    wire [47:0] p1;
    DSP48E2 #(.PREG(1)) dsp1 (
        .A     (27'd13),
        .B     ({2'b00, sample}),
        .C     (48'd0),
        .D     (27'd0),
        .OPMODE(9'h025),
        .ALUMODE(4'h0),
        .INMODE(5'h00),
        .CLK   (clk),
        .CEP   (~rst),
        .RSTP  (rst),
        .P     (p1)
    );

    // AIG glue : 48-bit add of the two DSP outputs (same-cycle macro -> AIG)
    assign mac = p0 + p1;

    // ---------------------------------------------------------------------
    //  CARRY4 x2 : 8-bit ripple adder   sum = sample[7:0] + addend
    //  s (select/propagate) = a ^ b ,  di (generate data) = a
    //  c_lo.CO[3] feeds c_hi.CI  ->  same-cycle CARRY4 -> CARRY4 edge
    // ---------------------------------------------------------------------
    wire [7:0] x    = sample[7:0];
    wire [3:0] s_lo = x[3:0] ^ addend[3:0];
    wire [3:0] s_hi = x[7:4] ^ addend[7:4];
    wire [3:0] o_lo, o_hi, co_lo, co_hi;

    CARRY4 c_lo (
        .DI    (x[3:0]),
        .S     (s_lo),
        .CI    (1'b0),
        .CYINIT(1'b0),
        .O     (o_lo),
        .CO    (co_lo)
    );

    CARRY4 c_hi (
        .DI    (x[7:4]),
        .S     (s_hi),
        .CI    (co_lo[3]),      // <-- direct macro->macro dependency, same cycle
        .CYINIT(1'b0),
        .O     (o_hi),
        .CO    (co_hi)
    );

    assign sum       = {o_hi, o_lo};
    assign sum_carry = co_hi[3];

endmodule
