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
// divmod.v: A simple divider/modulus component.  This implementation   //
//   is multi-cycle but high frequency.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// divmod_u - unsigned version

module divmod_u (reset, clk, go, n, d, q, r, done);
	parameter DATAWIDTH = 32;
	input reset, clk, go;
	input [DATAWIDTH-1:0] n, d;
	output [DATAWIDTH-1:0] q, r;
	inout done;
	reg running;
	wire start;
	reg [DATAWIDTH-1:0] qreg, nreg, res;
	wire [DATAWIDTH-1:0] partial;
	wire [DATAWIDTH:0] sub;
	reg [1:DATAWIDTH] iter;
	
	always @(posedge clk or posedge reset)
	begin
		if ( reset )
			running <= 0;
		else
		begin
			if ( start )
				running <= 1;
			else if ( done )
				running <= 0;
		end
	end
	
	assign start = go && !running;

	always @(posedge clk)
	begin
		if ( start )
			nreg <= n;
		else
			nreg <= {nreg[DATAWIDTH-2:0], 1'b0};
	end
	
	assign partial = {res[DATAWIDTH-2:0], nreg[DATAWIDTH-1]};
	assign sub = {1'b0, partial} - {1'b0, d};
	
	always @(posedge clk or posedge start)
	begin
		if ( start )
			res <= 0;
		else
			if ( !done )
			begin
				if ( !sub[DATAWIDTH] )
					res <= sub[DATAWIDTH-1:0];
				else
					res <= partial;
			end
	end
	
	always @(posedge clk)
	begin
		if ( !done )
			qreg <= {qreg[DATAWIDTH-2:0], !sub[DATAWIDTH]};
	end
	
	always @(posedge clk or posedge start)
	begin
		if ( start )
			iter <= 0;
		else
			iter <= {1'b1, iter[1:DATAWIDTH-1]};
	end
	
	assign q = qreg;
	assign r = res;
	assign done = iter[DATAWIDTH];
endmodule


// divmod_s - signed version

module divmod_s (reset, clk, go, n, d, q, r, done);
	parameter DATAWIDTH = 32;
	input reset, clk, go;
	input [DATAWIDTH-1:0] n, d;
	output [DATAWIDTH-1:0] q, r;
	inout done;
	wire [DATAWIDTH-2:0] qu, ru;
	
	divmod_u #(DATAWIDTH-1) divmodu(
		reset, clk, go, n[DATAWIDTH-2:0], d[DATAWIDTH-2:0], qu, ru, done
	);
	
	assign q = {n[DATAWIDTH-1] ^ d[DATAWIDTH-1], qu};
	assign r = {n[DATAWIDTH-1], ru};
endmodule

