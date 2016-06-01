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
// csignal.v: Implements the Impulse C signal.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module csignal (
	reset, clk, input_en, input_data, output_en, output_rdy, output_data
);
	parameter datawidth = 8;
	input reset, clk, input_en, output_en;
	input [datawidth-1:0] input_data;
	output output_rdy;
	output [datawidth-1:0] output_data;
	reg [datawidth-1:0] value;
	reg signaled;
	
	always @(posedge clk)
	begin
		if ( input_en )
			value <= input_data;
	end
	
	always @(posedge clk or posedge reset)
	begin
		if ( reset )
			signaled <= 0;
		else
			signaled <= input_en || (signaled && !output_en);
	end

	assign output_rdy = signaled;
	assign output_data = value;	
endmodule

module csignal_nodata (
    reset, clk, input_en, output_en, output_rdy
);
	input reset, clk, input_en, output_en;
	output output_rdy;
	reg signaled;

	always @(posedge reset or posedge clk)
	begin
		if ( reset )
			signaled <= 0;
		else
			signaled <= input_en || (signaled && !output_en);
	end

	assign output_rdy = signaled;
endmodule

