/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

// -----------------------------------------------------------------------------

void bli_sdotv_armv8a_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       float*  restrict rho,
       cntx_t* restrict cntx
     )
{
// tuning options:
#define UNROLL_BY_4
#undef  UNROLL_BY_8

#define N_ELEM_PER_REG 4
        const uint_64t /*dim_t*/  n_elem_per_reg = N_ELEM_PER_REG;   // using uint64_t w/interfacing with assembly; 
	                                                             // ensuring 64 bit.
#if     defined(UNROLL_BY_4)
#warning "UNROLL_BY_4 may not be complete"
#define N_ITER_UNROLL 4

#elif   defined(UNROLL_BY_8)
#warning "UNROLL_BY_8 may not be complete"
#define N_ITER_UNROLL 8
#endif

	const uint64_t /*dim_t*/  n_iter_unroll  = N_ITER_UNROLL;

#define UNROLL_STRIDE N_ITER_UNROLL*N_ELEM_PER_REG*sizeof(float)

	uint_64t /*dim_t*/ n_viter; 
	uint 64t /*dim_t*/ n_left;

	float            rho0; // scalar accumulator
	float           *rho0_addr = &rho0; // this is a conservative way to pass address to assembly program.

	// If the vector dimension is zero, set rho to zero and return early.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(s,set0s)( *rho );
		return;
	}

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}


#if 0
	// Initialize the local scalar rho1 to zero. --it says rho1, but you initialize rho0
	PASTEMAC(s,set0s)( rho0 );
#endif

	if(n_viter)
        {
	asm volatile
	(
	// set up pointers:
	 " ldr x0, %[xaddr]                           \n\t" // Load address of x vector.
	 " ldr x1, %[yaddr]                           \n\t" // Load address of y vector.

	// Initialize the unrolled iterations' rho vectors to zero.

	 " dup v24.4s, wzr                            \n\t" // v24 holds rho0; zero to start (wzr==0)
	 " dup v25.4s, wzr                            \n\t" // v25 holds rho1
	 " dup v26.4s, wzr                            \n\t" // v26 holds rho2
	 " dup v27.4s, wzr                            \n\t" // v27 holds rho3
#ifdef UNROLL_BY_8
	 " dup v28.4s, wzr                            \n\t" // v28 holds rho4
	 " dup v29.4s, wzr                            \n\t" // v29 holds rho5
	 " dup v30.4s, wzr                            \n\t" // v30 holds rho6
	 " dup v31.4s, wzr                            \n\t" // v31 holds rho7
#endif

	 // initialize loop:
	 " ldr x8, %[_n_viter]                        \n\t" // n_viter

	 ".LOOPITER:                                  \n\t" // main loop
	 " or x8, x8, x8                              \n\t" // 
	 " beq LOOPEND                                \n\t"

	 // float loads to vectors:
	 // "ldr q0, [x0]" and "ld1.4s {v0}, [x0]"... should be equivalent; but let's see why or not.

	 " ldr q0, [x0]                               \n\t" // 16 elements of x
	 " ldr q1, [x0, #16]                          \n\t"
	 " ldr q2, [x0, #32]                          \n\t"
	 " ldr q3, [x0, #48]                          \n\t"
#ifdef UNROLL_BY_8
	 " ldr q4, [x0, #64]                          \n\t" // 16 more elements of x
	 " ldr q5, [x0, #80]                          \n\t"
	 " ldr q6, [x0, #96]                          \n\t"
	 " ldr q7, [x0, #112]                         \n\t"
#endif
	 " ldr q16,[x1]                               \n\t" // 16 elements of y
	 " ldr q17,[x1, #16]                          \n\t"
	 " ldr q18,[x1, #32]                          \n\t"
	 " ldr q19,[x1, #48]                          \n\t"
#ifdef UNROLL_BY_8
	 " ldr q20,[x1, #64]                          \n\t" // 16 more elements of y
	 " ldr q21,[x1, #80]                          \n\t"
	 " ldr q22,[x1, #96]                          \n\t"
	 " ldr q23,[x1, #112]                         \n\t"
#endif
	 // possible prefetching here

	 " fmla v24.4s, v0.4s, v16.4s                 \n\t"
	 " fmla v25.4s, v1.4s, v17.4s                 \n\t"
	 " fmla v26.4s, v2.4s, v18.4s                 \n\t"
	 " fmla v27.4s, v3.4s, v19.4s                 \n\t"

	 // possible prefetching here

#ifdef UNROLL_BY_8 // is there really a danger in terms of pipeline of reusing accumulators too soon?
	 " fmla v28.4s, v4.4s, v20.4s                 \n\t"
	 " fmla v29.4s, v5.4s, v21.4s                 \n\t"
	 " fmla v30.4s, v6.4s, v22.4s                 \n\t"
	 " fmla v31.4s, v7.4s, v23.4s                 \n\t"
#endif

	 // update pointers:
	 " add x0, x0, #" ## #UNROLL_STRIDE ## "      \n\t"
	 " add x1, x1, #" ## #UNROLL_STRIDE ## "      \n\t"

	 // end of loop
	 " sub x8, x8, 1                              \n\t"
	 " b  .LOOPITER                               \n\t" 

	 ".LOOPEND:                                   \n\t"
	 //
	 // clean up the remainder of the work:
	 //
	 // 
	 " ldr x8, %[_n_left]                         \n\t" // n_left

#ifdef UNROLL_BY_8 
	 " tbz x8, #16, SKIP_FOR_VECTORS_AFTER        \n\t"

	 // consolidate to 4 accumulators here:
	 " fadd v24.4s, v28.4s, v24.4s                \n\t"
	 " fadd v25.4s, v29.4s, v25.4s                \n\t"
	 " fadd v26.4s, v30.4s, v26.4s                \n\t"
	 " fadd v27.4s, v31.4s, v27.4s                \n\t"

	 // four vector code if four vectors of work remain:

	 " ldr q0, [x0]                               \n\t" // 16 elements of x
	 " ldr q1, [x0, #16]                          \n\t"
	 " ldr q2, [x0, #32]                          \n\t"
	 " ldr q3, [x0, #48]                          \n\t"

	 " ldr q16,[x1]                               \n\t" // 16 elements of y
	 " ldr q17,[x1, #16]                          \n\t"
	 " ldr q18,[x1, #32]                          \n\t"
	 " ldr q19,[x1, #48]                          \n\t"

	 " fmla v24.4s, v0.4s, v16.4s                 \n\t"
	 " fmla v25.4s, v1.4s, v17.4s                 \n\t"
	 " fmla v26.4s, v2.4s, v18.4s                 \n\t"
	 " fmla v27.4s, v3.4s, v19.4s                 \n\t"

	 // update pointers:
	 " add x0, x0, #64                            \n\t"
	 " add x1, x1, #64                            \n\t"

	 ".SKIP_FOUR_VECTORS_AFTER:                   \n\t"
#endif
	 // consolidate to 2 accumulators here:
	 " fadd v24.4s, v26.4s, v24.4s                \n\t"
	 " fadd v25.4s, v27.4s, v25.4s                \n\t"

	 //two vector code if two vectors of work remain:
	 " tbz x8, #8, SKIP_TWO_VECTORS_AFTER         \n\t"

	 " ldr q0, [x0]                               \n\t" // 8 elements of x
	 " ldr q1, [x0, #16]                          \n\t"

	 " ldr q16,[x1]                               \n\t" // 8 elements of y
	 " ldr q17,[x1, #16]                          \n\t"

	 " fmla v24.4s, v0.4s, v16.4s                 \n\t"
	 " fmla v25.4s, v1.4s, v17.4s                 \n\t"

	 // update pointers:
	 " add x0, x0, #32                            \n\t"
	 " add x1, x1, #32                            \n\t"

	 ".SKIP_TWO_VECTORS_AFTER:                    \n\t"
	 // consolidate to 1 accumulator here:
	 " fadd v24.4s, v25.4s, v24.4s                \n\t"

	 //one vector code if one vector of work remains:
	 " tbz x8, #4, SKIP_ONE_VECTOR_AFTER          \n\t"

	 " ldr q0, [x0]                               \n\t" // 4 elements of x

	 " ldr q16,[x1]                               \n\t" // 4 elements of y

	 " fmla v24.4s, v0.4s, v16.4s                 \n\t"

	 // update pointers:
	 " add x0, x0, #16                            \n\t"
	 " add x1, x1, #16                            \n\t"

	 ".SKIP_ONE_VECTOR_AFTER:                     \n\t"

	 // consolidate to 1 accumulator here:
	 " fadd v24.4s, v25.4s, v24.4s                \n\t"

	 //one two floats of work remain?
	 " tbz x8, #2, SKIP_TWO_FLOATS_AFTER          \n\t"

	 " ldr d0, [x0]                              \n\t" // x: 2 floats must be loaded [or ld1.2s ]

	 " ldr d16,[x1]                              \n\t" // y: 2 floats must be loaded

	 "fmla v24.2s, v0.2s, v16.2s                  \n\t" // two fma operations needed     [bottom half of reg.]
	                                                    // (top two floats should be zeroed on write)
	 // update pointers:
	 " add x0, x0, #8                             \n\t"
	 " add x1, x1, #8                             \n\t"

	 ".SKIP_TWO_FLOATS_AFTER:                     \n\t"

	 //one two floats of work remain?
	 " tbz x8, #1, SKIP_ONE_FLOAT_AFTER           \n\t"

#warning "Finish post-vector code"
#ifndef APPROACH_1
	 " dup v0.4s, wzr                             \n\t" // zero
#endif
	 " dup v16.4s, wzr                            \n\t" //

	 " ld1 {v0.4s}[0], [x0]                       \n\t" // x: 1 float (same bits as v0.S)

	 " ld1 {v16.4s}[0],[x1]                       \n\t" // y: 1 float (same bits as v16.S)

#ifdef APPROACH_1
	 //this form exists: fmla Vd.S, Vd'.S, Vm.S[index]; but docs say 0<=m<=15. not 0<=m<=31.
	 " fmla v24.4s, v16.4s, {v0.4s}[0]            \n\t" // would be legal.
#else
	 " fmla v24.4s, v0.4s, v16.4s                 \n\t" // one fma operation needed
	                                                    // top three entries were pre-zeroed
#endif
	 // update pointers:
	 " add x0, x0, #4                             \n\t"
	 " add x1, x1, #4                             \n\t"

	 ".SKIP_ONE_FLOAT_AFTER:                      \n\t"

	 // consolidate accumulator in v24 to a scalar here (in v24.s[0]; same bits as v24.S):
	 "faddp v24.4s, v24.4s, v24.4s                \n\t" // reduction: [0]new := sum [0]+[1]; [1]new := sum [2] : [3]
	 "faddp v24.4s, v24.4s, v24.4s                \n\t" // reduction: [0]new := sum [0]+[1]==> sum of all 4 elements.

	 // store scalar float result:
	 " ldr x9,%[_rho0_addr]                       \n\t"
	 " st1 {v24.s}[0], [x9]                       \n\t" -- verified this instruction is right ; could use V24.S?
	 : // output operands (none)
	 : // input operands  (none)
           [xaddr]      "m" (x),       // 0
           [yaddr]      "m" (y),       // 1
	   [_n_viter]   "m" (n_viter)  // 2
	   [_n_left]    "m" (n_left)   // 3
	   [_rho0_addr] "m" (rho0_addr)// 4 --- alt here: [_rho0_addr] "r" (&rho0) 
	 : // Register clobber list
	   "x0", "x1", "x8", "x9"  // avoid x16-x18 and x29-x30; x19-x28 must be callee saved.
           "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
	 //"v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", // don't use these [v8-v15 are also callee saved if used].
	   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
	   "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"   // Note: less registers clobbered for unroll-by-4!
        );
	}
	else
	{
	   // Initialize local pointers.
   	   float*  restrict x0_ptr;
	   float*  restrict y0_ptr;

	   x0_ptr = x;
	   y0_ptr = y;

	   // If there are leftover iterations, perform them with scalar code.
	   for ( i = 0; i < n_left; ++i )
	   {
	      const float x0c = *x0_ptr;
	      const float y0c = *y0_ptr;

	      rho0 += x0c * y0c;

	      x0_ptr += incx;
	      y0_ptr += incy;
	   }
	}
#endif
	// Copy the final result into the output variable.
	PASTEMAC(s,copys)( rho0, *rho );
}

// -----------------------------------------------------------------------------

void bli_ddotv_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	double* restrict x0;
	double* restrict y0;
	double           rho0;

	v4df_t           rho0v, rho1v, rho2v, rho3v;
	v4df_t           x0v, y0v;
	v4df_t           x1v, y1v;
	v4df_t           x2v, y2v;
	v4df_t           x3v, y3v;

	// If the vector dimension is zero, set rho to zero and return early.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(d,set0s)( *rho );
		return;
	}

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Initialize the local scalar rho1 to zero.
	PASTEMAC(d,set0s)( rho0 );

	// Initialize the unrolled iterations' rho vectors to zero.
	rho0v.v = _mm256_setzero_pd();
	rho1v.v = _mm256_setzero_pd();
	rho2v.v = _mm256_setzero_pd();
	rho3v.v = _mm256_setzero_pd();

	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		x0v.v = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
		y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

		x1v.v = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
		y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

		x2v.v = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
		y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

		x3v.v = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
		y3v.v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rho0v.v = _mm256_fmadd_pd( x0v.v, y0v.v, rho0v.v );
		rho1v.v = _mm256_fmadd_pd( x1v.v, y1v.v, rho1v.v );
		rho2v.v = _mm256_fmadd_pd( x2v.v, y2v.v, rho2v.v );
		rho3v.v = _mm256_fmadd_pd( x3v.v, y3v.v, rho3v.v );

		x0 += ( n_elem_per_reg * n_iter_unroll );
		y0 += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
	rho0v.v += rho1v.v;
	rho0v.v += rho2v.v;
	rho0v.v += rho3v.v;

	// Accumulate the final rho vector into a single scalar result.
	rho0 += rho0v.d[0] + rho0v.d[1] + rho0v.d[2] + rho0v.d[3];

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const double x0c = *x0;
		const double y0c = *y0;

		rho0 += x0c * y0c;

		x0 += incx;
		y0 += incy;
	}

	// Copy the final result into the output variable.
	PASTEMAC(d,copys)( rho0, *rho );
}

