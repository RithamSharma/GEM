`timescale 1ns/1ps
module tb_sram;
    reg clk = 0;
    reg we = 0;
    reg [12:0] raddr = 0;
    reg [12:0] waddr = 0;
    reg [31:0] wdata = 0;
    wire [31:0] q;
    integer expected;
    reg [1023:0] vcd_path;
    reg [1023:0] expected_path;
    sram_top uut(.*);
    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("VCD=%s", vcd_path))
            vcd_path = "build/sram/oracle.vcd";
        if (!$value$plusargs("EXPECTED=%s", expected_path))
            expected_path = "build/sram/expected.csv";
        $dumpfile(vcd_path);
        $dumpvars(0, tb_sram);
        expected = $fopen(expected_path, "w");
        $fwrite(expected, "time,q\n");

        // Write address 7. The simultaneous read is intentionally ignored
        // because uninitialised HDL memory is X at this first edge.
        #2; we = 1; waddr = 7; raddr = 7; wdata = 32'hA5A5_5A5A;
        @(posedge clk); #1;
        we = 0;

        // Synchronous read observes the word written at the previous edge.
        @(posedge clk); #1;
        $fwrite(expected, "%0t,%08x\n", $time, q);

        // Replace it and prove read-before-write on the same address.
        we = 1; wdata = 32'hDEAD_BEEF;
        @(posedge clk); #1;
        $fwrite(expected, "%0t,%08x\n", $time, q);
        we = 0;
        @(posedge clk); #1;
        $fwrite(expected, "%0t,%08x\n", $time, q);

        $fclose(expected);
        // Give the VCD consumer a timestamp after the final active edge so it
        // can flush that cycle into GEM's output stream.
        // One extra inactive edge flushes GEM's documented two-timestamp VCD
        // pipeline; it is not part of the checked SRAM sequence.
        @(posedge clk);
        @(negedge clk);
        #1;
        $finish;
    end
endmodule
