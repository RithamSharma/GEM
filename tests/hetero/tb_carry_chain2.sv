`timescale 1ns/1ps

module tb_carry_chain2;
    reg         clk = 0;
    reg  [7:0] di;
    reg  [7:0] s;
    reg        ci;
    wire [7:0] o;
    wire [7:0] co;
    wire sampled_carry;
    integer i;
    integer bit_index;
    reg carry;
    reg [7:0] expected_o;
    reg [7:0] expected_co;

    reg [1023:0] vcd_path;
    reg [1023:0] expected_path;
    integer expected_file;

    carry_chain2 uut(
        .clk(clk), .di(di), .s(s), .ci(ci), .o(o), .co(co),
        .sampled_carry(sampled_carry)
    );

    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("VCD=%s", vcd_path))
            vcd_path = "build/carry-chain2/oracle.vcd";
        if (!$value$plusargs("EXPECTED=%s", expected_path))
            expected_path = "build/carry-chain2/expected.csv";
        $dumpfile(vcd_path);
        $dumpvars(0, tb_carry_chain2);
        expected_file = $fopen(expected_path, "w");
        if (expected_file == 0)
            $fatal(1, "cannot open expected output file");
        $fwrite(expected_file, "cycle,time,o,co\n");

        @(posedge clk);
        for (i = 0; i < 1024; i = i + 1) begin
            @(negedge clk);
            di = $urandom;
            s = $urandom;
            ci = $urandom;
            @(posedge clk);
            #1;
            carry = ci;
            expected_o = 0;
            expected_co = 0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                expected_o[bit_index] = s[bit_index] ^ carry;
                carry = s[bit_index] ? carry : di[bit_index];
                expected_co[bit_index] = carry;
            end
            if (o !== expected_o || co !== expected_co)
                $fatal(1, "chain mismatch di=%h s=%h ci=%b o=%h/%h co=%h/%h",
                       di, s, ci, o, expected_o, co, expected_co);
            $fwrite(expected_file, "%0d,%0d,%02h,%02h\n",
                    i, ($time - 1) * 1000, o, co);
        end
        $fclose(expected_file);
        $display("CARRY CHAIN PASS: 1024 vectors");
        $finish;
    end
endmodule
