module carry_only (
    input wire [3:0] di, s,
    input wire ci, cyinit,
    output wire [3:0] o, co
);
    CARRY4 carry (.DI(di), .S(s), .CI(ci), .CYINIT(cyinit), .O(o), .CO(co));
endmodule
