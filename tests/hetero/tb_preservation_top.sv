// SPDX-License-Identifier: Apache-2.0
// Long-run stimulus for preservation_top (1 DSP48E2 + 1 CARRY4 + 1 SRLC32E),
// used only as a Part D throughput fixture -- not a correctness oracle
// (tb_300.sv is the checked 300-cycle oracle for this design).
//
//   iverilog -g2012 -s tb_preservation_top -o pt.vvp \
//       tests/hetero/behavioral_zenith_macros.sv tests/hetero/preservation_top.sv \
//       tests/hetero/tb_preservation_top.sv
//   vvp pt.vvp +CYCLES=4000 +VCD=pt.vcd

module tb_preservation_top;
    reg         clk = 1'b0;
    reg  [26:0] a = 0, d = 0;
    reg  [17:0] b = 0;
    reg  [47:0] c = 0;
    reg  [3:0]  s = 0, di = 0;
    reg         ci = 0, cyinit = 0;
    reg         srl_d = 0, srl_ce = 0;
    reg  [4:0]  srl_a = 0;
    wire [47:0] p;
    wire [3:0]  o, co;
    wire        q, q31;

    preservation_top uut (
        .clk(clk), .a(a), .d(d), .b(b), .c(c),
        .s(s), .di(di), .ci(ci), .cyinit(cyinit),
        .srl_d(srl_d), .srl_ce(srl_ce), .srl_a(srl_a),
        .p(p), .o(o), .co(co), .q(q), .q31(q31)
    );

    integer k, ncyc;
    reg [8191:0] vcd;   // wide enough for an absolute path (iverilog %s fills from the LSB)
    initial begin
        if (!$value$plusargs("CYCLES=%d", ncyc)) ncyc = 4000;
        if (!$value$plusargs("VCD=%s", vcd))     vcd  = "preservation_top.vcd";
        $dumpfile(vcd);
        $dumpvars(0, tb_preservation_top);
        for (k = 0; k < ncyc; k = k + 1) begin
            a      = $random; d     = $random; b   = $random;
            c      = {$random, $random};
            s      = $random; di    = $random;
            ci     = $random; cyinit = $random;
            srl_d  = $random; srl_ce = $random; srl_a = $random;
            #5 clk = 1'b1;
            #5 clk = 1'b0;
        end
        $finish;
    end
endmodule
