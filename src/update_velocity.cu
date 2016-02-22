//-------------------------------------------------------------------------------
// Function : update_velocity
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Update fluid velocity.
//
// This function has 22 arguments.
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

__global__ void update_velocity( float *rho_d,  float *ux_d,   float *uy_d,
                                 float *uz_d,   float *g_1_d,  float *g_2_d,
                                 float *g_3_d,  float *g_4_d,  float *g_5_d,
                                 float *g_6_d,  float *g_7_d,  float *g_8_d,
                                 float *g_9_d,  float *g_10_d, float *g_11_d,
                                 float *g_12_d, float *g_13_d, float *g_14_d,
                                 float *g_15_d, float *g_16_d, float *g_17_d,
                                 float *g_18_d )
{
    int   i, idx, idxTmp, j, k;
    float invRho;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    idxTmp = i + NX * j;

    for( k = 1; k < NZ-1; k++ ){
        idx    = idxTmp + NXY * k;
        invRho = 1.0f / rho_d[idx];

        ux_d[idx] = ( g_1_d[idx]  - g_2_d[idx]  + g_7_d[idx]  - g_8_d[idx]
                  +   g_9_d[idx]  - g_10_d[idx] + g_11_d[idx] - g_12_d[idx]
                  +   g_13_d[idx] - g_14_d[idx] ) * invRho;

        uy_d[idx] = ( g_3_d[idx]  - g_4_d[idx]  + g_7_d[idx]  - g_8_d[idx]
                  -   g_9_d[idx]  + g_10_d[idx] + g_15_d[idx] - g_16_d[idx]
                  +   g_17_d[idx] - g_18_d[idx] ) * invRho;

        uz_d[idx] = ( g_5_d[idx]  - g_6_d[idx]  + g_11_d[idx] - g_12_d[idx]
                  -   g_13_d[idx] + g_14_d[idx] + g_15_d[idx] - g_16_d[idx]
                  -   g_17_d[idx] + g_18_d[idx] ) * invRho;

    }

}

