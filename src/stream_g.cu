//-------------------------------------------------------------------------------
// Function : stream_g
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Streaming step for the momentum distribution function g. 
//
// This function has 25 arguments.
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

__global__ void stream_g( int *nb_east_d, int *nb_west_d, 
                          int *nb_north_d,   int *nb_south_d,
                          float *g_0_d,  float *g_1_d,  float *g_2_d, 
                          float *g_3_d,  float *g_4_d,  float *g_5_d,
                          float *g_6_d,  float *g_7_d,  float *g_8_d,
                          float *g_9_d,  float *g_10_d, float *g_11_d,
                          float *g_12_d, float *g_13_d, float *g_14_d, 
                          float *g_15_d, float *g_16_d, float *g_17_d,
                          float *g_18_d )
{
    int i, idx, idx2, ie, iw, j, jn, js, k, kt, kb;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    for( k = 1; k < NZ-1; k++ ){

        idx  = gridId( i, j, k );
        idx2 = gridId( i, j, k ) + dcol;

        ie = nb_east_d[idx];
        iw = nb_west_d[idx];
        jn = nb_north_d[idx];
        js = nb_south_d[idx];
        kt = k + 1;
        kb = k - 1;

        g_0_d[idx] = g_0_d[idx2];

        g_1_d[ gridId(ie,j,k) ] = g_1_d[idx2];
        g_2_d[ gridId(iw,j,k) ] = g_2_d[idx2];
        g_3_d[ gridId(i,jn,k) ] = g_3_d[idx2];
        g_4_d[ gridId(i,js,k) ] = g_4_d[idx2];
        g_5_d[ gridId(i,j,kt) ] = g_5_d[idx2];
        g_6_d[ gridId(i,j,kb) ] = g_6_d[idx2];

        g_7_d[ gridId(ie,jn,k) ]  = g_7_d[idx2];
        g_8_d[ gridId(iw,js,k) ]  = g_8_d[idx2];
        g_9_d[ gridId(ie,js,k) ]  = g_9_d[idx2];
        g_10_d[ gridId(iw,jn,k) ] = g_10_d[idx2];
        g_11_d[ gridId(ie,j,kt) ] = g_11_d[idx2];
        g_12_d[ gridId(iw,j,kb) ] = g_12_d[idx2];
        g_13_d[ gridId(ie,j,kb) ] = g_13_d[idx2];
        g_14_d[ gridId(iw,j,kt) ] = g_14_d[idx2];
        g_15_d[ gridId(i,jn,kt) ] = g_15_d[idx2];
        g_16_d[ gridId(i,js,kb) ] = g_16_d[idx2];
        g_17_d[ gridId(i,jn,kb) ] = g_17_d[idx2];
        g_18_d[ gridId(i,js,kt) ] = g_18_d[idx2];

    }

}



