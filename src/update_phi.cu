//-------------------------------------------------------------------------------
// Function : update_phi
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Update order parameter.
//
// This function has 8 arguments.
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

__global__ void update_phi( float *phi_d, float *f_0_d, float *f_1_d,
                            float *f_2_d, float *f_3_d, float *f_4_d,
                            float *f_5_d, float *f_6_d )
{
    int i, idx, j, k, idxTmp;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    idxTmp = i + NX * j;

    // Order parameter update
    for( k = 1; k < NZ-1; k++ ){
        idx = idxTmp + NXY * k;
        phi_d[idx] = f_0_d[idx] + f_1_d[idx] + f_2_d[idx] + f_3_d[idx]
                   + f_4_d[idx] + f_5_d[idx] + f_6_d[idx];
    }

}

