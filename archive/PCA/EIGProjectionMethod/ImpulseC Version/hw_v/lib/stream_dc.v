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
// stream_dc.v: Wrapper for fifo_sync_dc module to interface with       //
//   CoBuilder-generated modules.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module stream_dc(
	ireset, iclk, input_en, input_rdy, input_eos, input_data,
	oreset, oclk, output_en, output_rdy, output_eos, output_data);

	parameter datawidth = 8;
	parameter addrwidth = 4;
	
	input ireset, iclk;
	input input_en;
	output input_rdy;
	input input_eos;
	input [datawidth-1:0] input_data;
	input oreset, oclk;
	input output_en;
	output output_rdy;
	output output_eos;
	output [datawidth-1:0] output_data;

	wire empty, full;
	wire [datawidth:0] fifo_in, fifo_out;
	wire rd_fifo;
	reg reading, available;
	reg [datawidth:0] oreg;
	wire [datawidth:0] dout;

	assign input_rdy = !full;
	assign fifo_in = {input_eos, input_data};
	assign output_data = dout[datawidth-1:0];
	assign output_eos = dout[datawidth] && available;

	fifo_sync_dc #(datawidth+1, addrwidth) fifo_1(oreset, oclk, ireset, iclk, 
  		rd_fifo, input_en, fifo_in, empty, full, fifo_out);

	//
	// Simulate asynchronous output by prefetching FIFO data.
	//

	// If the FIFO is not empty and we don't have any data (i.e., not available)
	// or its being consumed (i.e., output_en) then read the next value.
	assign rd_fifo = !empty && (!available || output_en);

	always @(posedge oreset or posedge oclk)
	begin
		if ( oreset )
			reading <= 0;
		else
			reading <= rd_fifo;
	end

	// Capture output from FIFO
	always @(posedge oclk)
	begin
		if ( reading )
			oreg <= fifo_out;
	end

	assign dout = reading ? fifo_out : oreg;

	always @(posedge oreset or posedge oclk)
	begin
		if ( oreset )
			available <= 0;
		else
			available <= rd_fifo || (available && !output_en);
	end

	assign output_rdy = available;

endmodule

