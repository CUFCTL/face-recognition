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
// fifo_dc.v: Generic synthesizable dual-clock FIFO.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

module fifo_sync_dc (
	r_reset, rclk, w_reset, wclk, read, write, din, empty, full, dout
);
	parameter DATAWIDTH = 8;
	parameter ADDRESSWIDTH = 4;
	parameter DEPTH = 1 << ADDRESSWIDTH;
	input r_reset, rclk, w_reset, wclk, read, write;
	input [DATAWIDTH-1:0] din;
	output [DATAWIDTH-1:0] dout;
	inout empty, full;

	reg empty_reg, full_reg;
	reg [DATAWIDTH-1:0] mem[DEPTH-1:0];
	// Write-side signals
	reg [ADDRESSWIDTH-1:0] in_ptr, in_ptr_gray, out_ptr_gray1, out_ptr_gray2;
	wire [ADDRESSWIDTH-1:0] next_in_ptr;
	reg [ADDRESSWIDTH-1:0] read_addra;
	wire [DATAWIDTH-1:0] doa;
	// Read-side signals
	reg [ADDRESSWIDTH-1:0] out_ptr, out_ptr_gray, in_ptr_gray1, in_ptr_gray2;
	wire [ADDRESSWIDTH-1:0] next_out_ptr;
	reg [ADDRESSWIDTH-1:0] read_addrb;
	wire [DATAWIDTH-1:0] dob;

	// Convert binary vector to gray code vector
	function [ADDRESSWIDTH-1:0] to_gray;
	input [ADDRESSWIDTH-1:0] vec;
	integer i;
	begin
		to_gray[ADDRESSWIDTH-1] = vec[ADDRESSWIDTH-1];
		for ( i = ADDRESSWIDTH - 2; i >= 0; i = i - 1 )
			to_gray[i] = vec[i] ^ vec[i + 1];
	end
	endfunction
  
	// Write side
	assign next_in_ptr = in_ptr + 1;
	assign full = full_reg;

	always @(posedge w_reset or posedge wclk)
	begin: writeptr
		if ( w_reset )
		begin
			in_ptr <= 0;
			in_ptr_gray <= 0;
			full_reg <= 0;
		end
		else
		begin
			full_reg <= (to_gray(next_in_ptr) == out_ptr_gray2);
			if ( write && !full )
			begin
				in_ptr <= next_in_ptr;
				in_ptr_gray <= to_gray(next_in_ptr);
			end
		end
	end

	// Synchronize out_ptr_gray
	always @(posedge w_reset or posedge wclk)
	begin
		if ( w_reset )
		begin
			out_ptr_gray1 <= 0;
			out_ptr_gray2 <= 0;
		end
		else
		begin
			out_ptr_gray1 <= out_ptr_gray;
			out_ptr_gray2 <= out_ptr_gray1;
		end
	end

	// Read side
	assign next_out_ptr = out_ptr + 1;
	assign empty = empty_reg;
	
	always @(posedge r_reset or posedge rclk)
	begin: readptr
		if ( r_reset )
		begin
			out_ptr <= 0;
			out_ptr_gray <= 0;
			empty_reg <= 0;
		end
		else
		begin
			empty_reg <= (in_ptr_gray2 == to_gray(out_ptr));
			if ( read && !empty )
			begin
				out_ptr <= next_out_ptr;
				out_ptr_gray <= to_gray(next_out_ptr);
			end
		end
	end
	
	assign dout = dob;
	
	// Inferred block RAM
	always @(posedge wclk)
	begin: writemem
		if ( write && !full )
			mem[in_ptr] <= din;
		read_addra <= in_ptr;
	end

	always @(posedge rclk)
	begin: readmem
		read_addrb <= out_ptr;
	end

	assign doa = mem[read_addra];
	assign dob = mem[read_addrb];

	// Synchronize in_ptr_gray
	always @(posedge r_reset or posedge rclk)
	begin
		if ( r_reset )
		begin
			in_ptr_gray1 <= 0;
			in_ptr_gray2 <= 0;
		end
		else
		begin
			in_ptr_gray1 <= in_ptr_gray;
			in_ptr_gray2 <= in_ptr_gray1;
		end
	end
endmodule

