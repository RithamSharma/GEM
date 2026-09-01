`timescale 1ns/1ps

module tb_300;
    reg clk = 1'b0;
    reg [26:0] a = 0;
    reg [26:0] d = 0;
    reg [17:0] b = 0;
    reg [47:0] c = 0;
    reg [3:0] s = 0;
    reg [3:0] di = 0;
    reg ci = 0;
    reg cyinit = 0;
    reg srl_d = 0;
    reg srl_ce = 0;
    reg [4:0] srl_a = 0;
    wire [47:0] p;
    wire [3:0] o;
    wire [3:0] co;
    wire q;
    wire q31;

    integer seed = 32'h5eed_3001;
    integer cycle;
    integer expected_file;
    reg [1023:0] vcd_path;
    reg [1023:0] expected_path;

    preservation_top uut (
        .clk(clk), .a(a), .d(d), .b(b), .c(c),
        .s(s), .di(di), .ci(ci), .cyinit(cyinit),
        .srl_d(srl_d), .srl_ce(srl_ce), .srl_a(srl_a),
        .p(p), .o(o), .co(co), .q(q), .q31(q31)
    );

    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("VCD=%s", vcd_path))
            vcd_path = "build/test300/oracle.vcd";
        if (!$value$plusargs("EXPECTED=%s", expected_path))
            expected_path = "build/test300/expected.csv";
        $dumpfile(vcd_path);
        $dumpvars(0, tb_300);
        expected_file = $fopen(expected_path, "w");
        if (expected_file == 0)
            $fatal(1, "cannot open expected output file");
        $fwrite(expected_file, "cycle,time,p,o,co,q,q31\n");

        // The initial posedge establishes the same zero-initialized P state
        // used by GEM. Every subsequent posedge is one checked vector.
        @(posedge clk);
        for (cycle = 0; cycle < 300; cycle = cycle + 1) begin
            @(negedge clk);
            a = $urandom(seed);
            d = $urandom(seed);
            b = $urandom(seed);
            c = {$urandom(seed), $urandom(seed)};
            s = $urandom(seed);
            di = $urandom(seed);
            ci = $urandom(seed);
            cyinit = $urandom(seed);
            srl_d = $urandom(seed);
            srl_ce = $urandom(seed);
            srl_a = $urandom(seed);
            @(posedge clk);
            #1;
            if ((^p === 1'bx) || (^o === 1'bx) || (^co === 1'bx) ||
                (q === 1'bx) || (q31 === 1'bx)) begin
                $display("unknown diagnostic p=%h o=%h co=%h q=%b q31=%b", p, o, co, q, q31);
                $fatal(1, "oracle produced X/Z at cycle %0d", cycle);
            end
            // The VCD uses the 1 ps precision unit, while $time is expressed
            // in the module's 1 ns unit.
            $fwrite(expected_file, "%0d,%0d,%012h,%01h,%01h,%0d,%0d\n",
                    cycle, ($time - 1) * 1000, p, o, co, q, q31);
        end
        $fclose(expected_file);
        $display("ORACLE PASS: generated 300 deterministic vectors");
        $finish;
    end
endmodule
