//-------------------------------------------------------------------------------
// Function : stream_f
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Streaming step for the order parameter distribution function f. 
//
// This function has 11 arguments.
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

__global__ void stream_f( int *nb_east_d,  int *nb_west_d, 
                          int *nb_north_d, int *nb_south_d, 
                          float *f_0_d, float *f_1_d, float *f_2_d, 
                          float *f_3_d, float *f_4_d, float *f_5_d, 
                          float *f_6_d )
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

        f_0_d[idx] = f_0_d[idx2];

        f_1_d[ gridId(ie,j,k) ] = eta_d*f_1_d[idx2] + eta2_d*f_1_d[ gridId(ie,j,k) + dcol ];
        f_2_d[ gridId(iw,j,k) ] = eta_d*f_2_d[idx2] + eta2_d*f_2_d[ gridId(iw,j,k) + dcol ];
        f_3_d[ gridId(i,jn,k) ] = eta_d*f_3_d[idx2] + eta2_d*f_3_d[ gridId(i,jn,k) + dcol ];
        f_4_d[ gridId(i,js,k) ] = eta_d*f_4_d[idx2] + eta2_d*f_4_d[ gridId(i,js,k) + dcol ];
        f_5_d[ gridId(i,j,kt) ] = eta_d*f_5_d[idx2] + eta2_d*f_5_d[ gridId(i,j,kt) + dcol ];
        f_6_d[ gridId(i,j,kb) ] = eta_d*f_6_d[idx2] + eta2_d*f_6_d[ gridId(i,j,kb) + dcol ];

    }

}



