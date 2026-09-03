// SPDX-License-Identifier: Apache-2.0
// Stimulus for bench_mac: toggles the clock and drives pseudo-random inputs for
// +CYCLES cycles, dumping a VCD. Used only to give the GEM simulators a long
// enough run to time (Part D). Not a correctness oracle.
//
//   iverilog -g2012 -s tb_bench_mac -o bm.vvp \
//       tests/hetero/behavioral_zenith_macros.sv tests/hetero/bench_mac.sv tests/hetero/tb_bench_mac.sv
//   vvp bm.vvp +CYCLES=4000 +VCD=bench_mac.vcd

module tb_bench_mac;
    reg         clk = 1'b0;
    reg  [26:0] xin = 27'b0;
    reg  [17:0] win = 18'b0;
    wire [47:0] acc0;
    wire [31:0] sum;

    bench_mac uut (.clk(clk), .xin(xin), .win(win), .acc0(acc0), .sum(sum));

    integer k, ncyc;
    reg [8191:0] vcd;   // wide enough for an absolute path (iverilog %s fills from the LSB)
    initial begin
        if (!$value$plusargs("CYCLES=%d", ncyc)) ncyc = 4000;
        if (!$value$plusargs("VCD=%s", vcd))     vcd  = "bench_mac.vcd";
        $dumpfile(vcd);
        $dumpvars(0, tb_bench_mac);
        for (k = 0; k < ncyc; k = k + 1) begin
            xin = $random;
            win = $random;
            #5 clk = 1'b1;
            #5 clk = 1'b0;
        end
        $finish;
    end
endmodule
