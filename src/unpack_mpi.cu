//-------------------------------------------------------------------------------
// Function : unpack_mpi
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Gathers MPI boundary data into two arrays only to reduce communication time 
// betweeen host and device. Data saved to real nodes.
//
// This function has 12 arguments.
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// Copyright 2016 Carlos Rosales Fernandez and The University of Texas at Austin.
// Copyright 2008 Carlos Rosales Fernandez, David S. Whyte and IHPC (A*STAR).
//
// This file is part of MP-LABS.
//
// MP-LABS is free software: you can redistribute it and/or modify it under the
// terms of the GNU GPL version 3 or (at your option) any later version.
//
// MP-LABS is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// MP-LABS, in the file COPYING.txt. If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------

__global__ void unpack_mpi( float *top_rcv_d, float *bot_rcv_d,
                            float *f_5_d,  float *f_6_d,  float *g_5_d,
                            float *g_6_d,  float *g_11_d, float *g_12_d,
                            float *g_13_d, float *g_14_d, float *g_15_d,
                            float *g_16_d, float *g_17_d, float *g_18_d )
{
    int   i, idx, idx2, j;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    // Inward pointing values for real nodes at the top boundary
    idx  = gridId( i, j, NZ-2 );
    idx2 = ( i + NX*j )*6;
    f_6_d[ idx ]  = top_rcv_d[   idx2 ];
    g_6_d[ idx ]  = top_rcv_d[ ++idx2 ];
    g_12_d[ idx ] = top_rcv_d[ ++idx2 ];
    g_13_d[ idx ] = top_rcv_d[ ++idx2 ];
    g_16_d[ idx ] = top_rcv_d[ ++idx2 ];
    g_17_d[ idx ] = top_rcv_d[ ++idx2 ];

    // Inward pointing values for real nodes at the bottom boundary
    idx  = gridId( i, j, 1 );
    idx2 = ( i + NX*j )*6;
    f_5_d[ idx ]  = bot_rcv_d[   idx2 ];
    g_5_d[ idx ]  = bot_rcv_d[ ++idx2 ];
    g_11_d[ idx ] = bot_rcv_d[ ++idx2 ];
    g_14_d[ idx ] = bot_rcv_d[ ++idx2 ];
    g_15_d[ idx ] = bot_rcv_d[ ++idx2 ];
    g_18_d[ idx ] = bot_rcv_d[ ++idx2 ];

}
