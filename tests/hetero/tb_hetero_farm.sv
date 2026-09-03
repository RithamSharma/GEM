// SPDX-License-Identifier: Apache-2.0
// Long-run pseudo-random stimulus for hetero_farm (Part D throughput fixture).
// Not a correctness oracle -- the 300-cycle tb_300 flow is the checked oracle
// for these same macro configs.
//
//   iverilog -g2012 -s tb_hetero_farm -o hf.vvp \
//       tests/hetero/behavioral_zenith_macros.sv tests/hetero/hetero_farm.sv \
//       tests/hetero/tb_hetero_farm.sv
//   vvp hf.vvp +CYCLES=4000 +VCD=hf.vcd

module tb_hetero_farm;
    localparam integer N = 32;      // keep in sync with hetero_farm's default

    reg         clk = 1'b0;
    reg  [26:0] a = 0, d = 0;
    reg  [17:0] b = 0;
    reg  [47:0] c = 0;
    reg  [3:0]  s = 0, di = 0;
    reg         ci = 0, cyinit = 0;
    reg         srl_d = 0, srl_ce = 0;
    reg  [4:0]  srl_a = 0;
    wire [48*N-1:0] p_all;
    wire [8*N-1:0]  oco_all;
    wire [2*N-1:0]  q_all;

    hetero_farm #(.N(N)) uut (
        .clk(clk), .a(a), .d(d), .b(b), .c(c),
        .s(s), .di(di), .ci(ci), .cyinit(cyinit),
        .srl_d(srl_d), .srl_ce(srl_ce), .srl_a(srl_a),
        .p_all(p_all), .oco_all(oco_all), .q_all(q_all)
    );

    integer k, ncyc;
    reg [8191:0] vcd;   // wide enough for an absolute path
    initial begin
        if (!$value$plusargs("CYCLES=%d", ncyc)) ncyc = 4000;
        if (!$value$plusargs("VCD=%s", vcd))     vcd  = "hetero_farm.vcd";
        $dumpfile(vcd);
        $dumpvars(0, tb_hetero_farm);
        for (k = 0; k < ncyc; k = k + 1) begin
            a      = $random; d      = $random; b   = $random;
            c      = {$random, $random};
            s      = $random; di     = $random;
            ci     = $random; cyinit = $random;
            srl_d  = $random; srl_ce = $random; srl_a = $random;
            #5 clk = 1'b1;
            #5 clk = 1'b0;
        end
        $finish;
    end
endmodule
