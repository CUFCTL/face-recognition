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
// fifo.v: Generic synthesizable FIFO                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module fifo (reset, clk, read, write, din, empty, full, dout);
	parameter DATAWIDTH = 8;
	parameter ADDRESSWIDTH = 4;
	parameter DEPTH = 1 << ADDRESSWIDTH;

	input reset, clk, read, write;
	input [DATAWIDTH-1:0] din;
	output [DATAWIDTH-1:0] dout;
	inout empty, full;

	reg [DATAWIDTH-1:0] mem[DEPTH-1:0];

	wire [ADDRESSWIDTH-1:0] next_ptr;
	reg [ADDRESSWIDTH-1:0] in_ptr, out_ptr;

	assign next_ptr = in_ptr + 1;

	always @(posedge clk)
	begin: writemem
		if ( write && !full )
			mem[in_ptr] <= din;
	end

	always @(posedge reset or posedge clk)
	begin: writeptr
		if ( reset )
			in_ptr <= 0;
		else
		begin
			if ( write && !full )
				in_ptr <= next_ptr;
		end
	end

	always @(posedge reset or posedge clk)
	begin: readptr
		if ( reset )
			out_ptr <= 0;
		else
		begin
			if ( read && !empty )
				out_ptr <= out_ptr + 1;
		end
	end

	assign dout = mem[out_ptr];

	assign full = (next_ptr == out_ptr);
	assign empty = (in_ptr == out_ptr);
endmodule

