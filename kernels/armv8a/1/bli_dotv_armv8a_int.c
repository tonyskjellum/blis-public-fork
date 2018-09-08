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

/* Union data structure to access ARM fp registers
   One 128-bit fp register holds 4 SP elements. */
typedef union
{
  //	__m256  v;
#warning "TBD".

	float   f[4] __attribute__((aligned(64)));
} v4sf_t;

/* Union data structure to access ARM fp registers
*  One 128-bit fp register holds 2 DP elements. */
typedef union
{
  //	__m256d v;
#warning "TBD."
	double  d[2] __attribute__((aligned(64)));

} v2df_t;

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

	const uint64_t /*dim_t*/  n_iter_unroll  = 4; // Revisit when optimizing; code is written this way now.

#elif   defined(UNROLL_BY_8)
#warning "UNROLL_BY_8 may not be complete"
#define N_ITER_UNROLL 8

	const uint64_t /*dim_t*/  n_iter_unroll  = N_ITER_UNROLL;

#endif
#define UNROLL_STRIDE N_ITER_UNROLL*N_ELEM_PER_REG*sizeof(float)

	uint_64t /*dim_t*/ n_viter; 
	uint 64t /*dim_t*/ n_left;

#if 0
	float*  restrict x0;
	float*  restrict y0;
#endif
	float            rho0;

// may not need explicit variables:
//	v4sf_t           rho0v, rho1v, rho2v, rho3v;
//
//      v4sf_t           x0v, y0v;
//      v4sf_t           x1v, y1v;
//      v4sf_t           x2v, y2v;
//      v4sf_t           x3v, y3v;
//

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

	// Initialize local pointers.
#if 0
	x0 = x;
	y0 = y;
#endif

#if 0
	// Initialize the local scalar rho1 to zero. --it says rho1, but you initialize rho0
	PASTEMAC(s,set0s)( rho0 );
#endif

	// Initialize the unrolled iterations' rho vectors to zero.
#if 0
	rho0v.v = _mm256_setzero_ps();
	rho1v.v = _mm256_setzero_ps();
	rho2v.v = _mm256_setzero_ps();
	rho3v.v = _mm256_setzero_ps();
#endif

	asm volatile
	(
	// set up pointers:
	 " ldr x0,%[xaddr]                            \n\t" // Load address of x vector.
	 " ldr x1,%[yaddr]                            \n\t" // Load address of y vector.
	 
	// Initialize the unrolled iterations' rho vectors to zero.

	 " dup v24.4s, wzr                            \n\t" // v24 holds rho0; zero to start
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
	 " ldr x8, %[_nv_iter]                        \n\t" // n_viter
	 " mov x9, x8                                 \n\t" // copy
#ifdef UNROLL_BY_4
	 " and x9, x9, #3                             \n\t" // remainder after unrolling by 4 vectors
#else
	 " and x9, x9, #7                             \n\t" // remainder after unrolling by 8 vectors
#endif
	 " sub x8, x8, x9                             \n\t" // remove tail work

	 ".LOOPITER:                                  \n\t" // main loop
	 " or x8, x8, x8                              \n\t" // 
	 " beq LOOPEND                                \n\t"

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

	 " fmla v24.4s, v0.4s, v4.4s                  \n\t"
	 " fmla v25.4s, v1.4s, v5.4s                  \n\t"
	 " fmla v26.4s, v2.4s, v6.4s                  \n\t"
	 " fmla v27.4s, v3.4s, v7.4s                  \n\t"

	 // possible prefetching here

#ifdef UNROLL_BY_8 // is there really a danger in terms of pipeline of reusing these accumulators?
	 " fmla v28.4s, v4.4s, v8.4s                  \n\t"
	 " fmla v29.4s, v5.4s, v9.4s                  \n\t"
	 " fmla v30.4s, v6.4s, v10.4s                 \n\t"
	 " fmla v31.4s, v7.4s, v11.4s                 \n\t"
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
#ifdef UNROLL_BY_8 
	 // consolidate to 4 accumulators here:

	 // four vector code if four vectors of work remain:

#endif
	 //two vector code if two vectors of work remain:

	 //one vector code if one vector of work remains:

	 //scalar code if needed:


	 : // output operands (none)
	 : // input operands  (none)
           [xaddr]    "m" (x),      // 0
           [yaddr]    "m" (y),      // 1
	   [_nv_iter] "m" (n_viter) // 2

	 : // Register clobber list
	   "x0", "x1", "x8",
           "v28", "v29", "v30", "v31"  // more coming soon 
        );


	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		x0v.v = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
		y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

		x1v.v = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
		y1v.v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

		x2v.v = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
		y2v.v = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );

		x3v.v = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
		y3v.v = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rho0v.v = _mm256_fmadd_ps( x0v.v, y0v.v, rho0v.v );
		rho1v.v = _mm256_fmadd_ps( x1v.v, y1v.v, rho1v.v );
		rho2v.v = _mm256_fmadd_ps( x2v.v, y2v.v, rho2v.v );
		rho3v.v = _mm256_fmadd_ps( x3v.v, y3v.v, rho3v.v );

		x0 += ( n_elem_per_reg * n_iter_unroll );
		y0 += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
#if 0
	rho0v.v += rho1v.v;
	rho0v.v += rho2v.v;
	rho0v.v += rho3v.v;
#endif

	// Accumulate the final rho vector into a single scalar result.
#if 0
	rho0 += rho0v.f[0] + rho0v.f[1] + rho0v.f[2] + rho0v.f[3] +
	        rho0v.f[4] + rho0v.f[5] + rho0v.f[6] + rho0v.f[7];
#endif

	// If there are leftover iterations, perform them with scalar code.
#if 0
	for ( i = 0; i < n_left; ++i )
	{
		const float x0c = *x0;
		const float y0c = *y0;

		rho0 += x0c * y0c;

		x0 += incx;
		y0 += incy;
	}
#endif
	// we will need to communicate rho0star as an out parameter and then let it paste here

	// Copy the final result into the output variable.
//	PASTEMAC(s,copys)( rho0, *rho );
	PASTEMAC(s,copys)( rho0star, *rho );
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

