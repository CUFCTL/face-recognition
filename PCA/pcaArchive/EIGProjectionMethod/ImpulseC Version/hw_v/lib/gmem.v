//////////////////////////////////////////////////////////////////////////
// Copyright (c) 2002-2007 by Impulse Accelerated Technologies, Inc.    //
// All rights reserved.                                                 //
//                                                                      //
// This source file may be used and redistributed without charge        //
// subject to the provisions of the IMPULSE ACCELERATED TECHNOLOGIES,   //
// INC. REDISTRIBUTABLE IP LICENSE AGREEMENT, and provided that this    //
// copyright statement is not removed from the file, and that any       //
// derivative work contains this copyright notice.                      //
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// gmem.v: Generic synthesizable RAM.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module gmem (rst, clk, we, addr, din, dout);
	parameter DATAWIDTH = 32;
	parameter ADDRWIDTH = 256;
	parameter SIZE = 2 << ADDRWIDTH - 1;
	input rst, clk, we;
	input [ADDRWIDTH-1:0] addr;
	input [DATAWIDTH-1:0] din;
	output [DATAWIDTH-1:0] dout;
	reg [DATAWIDTH-1:0] mem[0:SIZE-1];

	assign dout = mem[addr];

	always @(posedge clk)
	begin
		if ( we )
			mem[addr] <= din;
	end
endmodule

