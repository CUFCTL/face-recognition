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
// cregister.v: Implements the Impulse C register.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module cregister (clk, input_en, input_data, output_data);
	parameter datawidth = 8;
	input clk, input_en;
	input [datawidth-1:0] input_data;
	output [datawidth-1:0] output_data;
	reg [datawidth-1:0] value;

	always @(posedge clk)
	begin
		if ( input_en )
			value <= input_data;
	end
	
	assign output_data = value;
endmodule

