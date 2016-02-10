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
// stream.v: Wrapper for fifo module to interface with CoBuilder-       //
//   generated modules.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module stream (
	reset, clk, input_en, input_rdy, input_eos, input_data,
	output_en, output_rdy, output_eos, output_data
);
	parameter datawidth = 8;
	parameter addrwidth = 4;
	input reset, clk, input_en, input_eos, output_en;
	input [datawidth-1:0] input_data;
	output input_rdy, output_rdy, output_eos;
	output [datawidth-1:0] output_data;

	wire empty, full;
	wire [datawidth:0] fifo_in, fifo_out;

	assign output_rdy = !empty;
	assign input_rdy = !full;
	assign fifo_in = {input_eos, input_data};
	assign output_data = fifo_out[datawidth-1:0];
	assign output_eos = fifo_out[datawidth] && !empty;

	fifo #(datawidth+1, addrwidth) fifo_1(reset, clk, output_en, input_en, 
		fifo_in, empty, full, fifo_out);
endmodule

