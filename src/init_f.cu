//-------------------------------------------------------------------------------
// Function : init_f
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Initialize all variables and arrays.
// This function requires 15 arguments.
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

__global__ void init_f( int *nb_east_d,  int *nb_west_d,
                        int *nb_north_d, int *nb_south_d,
                        float *phi_d,    float *f_0_d,
                        float *f_1_d,    float *f_2_d,
                        float *f_3_d,    float *f_4_d,
                        float *f_5_d,    float *f_6_d )
{
    int   i, idx, ie, iw, j, jn, js, k, kt, kb;
    float Af, muPhin, phin, lapPhi;

    // Identify current thread
	  i = blockIdx.x * blockDim.x + threadIdx.x;
	  j = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize distribution function f
    for( k = 1; k < NZ-1; k++ ){

        // Define some local values
        idx  = gridId( i, j, k );
        phin = phi_d[idx];

        // Differential terms
        ie = nb_east_d[idx];
        iw = nb_west_d[idx];
        jn = nb_north_d[idx];
        js = nb_south_d[idx];
        kt = k + 1;
        kb = k - 1;

        // Laplacian of the order parameter Phi
        lapPhi = ( phi_d[ gridId(ie,jn,k ) ] + phi_d[ gridId(iw,js,k ) ] 
               +   phi_d[ gridId(ie,js,k ) ] + phi_d[ gridId(iw,jn,k ) ]
               +   phi_d[ gridId(ie,j ,kt) ] + phi_d[ gridId(iw,j ,kb) ]
               +   phi_d[ gridId(ie,j ,kb) ] + phi_d[ gridId(iw,j ,kt) ]
               +   phi_d[ gridId(i ,jn,kt) ] + phi_d[ gridId(i ,js,kb) ]
               +   phi_d[ gridId(i ,jn,kb) ] + phi_d[ gridId(i ,js,kt) ]
               + 2.f*( phi_d[ gridId(ie,j ,k ) ] + phi_d[ gridId(iw,j ,k ) ] 
               +       phi_d[ gridId(i ,jn,k ) ] + phi_d[ gridId(i ,js,k ) ] 
               +       phi_d[ gridId(i ,j ,kt) ] + phi_d[ gridId(i ,j ,kb) ]
               -       12.f*phin ) )*inv6;

         // Chemical potential
         muPhin = alpha4_d*phin*( phin*phin - phiStarSq_d ) - kappa_d*lapPhi;

        // Set distribution function f to its equilibrium value
        Af = 0.5f*Gamma_d*muPhin;

        f_0_d[idx] = phin - 3.f*Gamma_d*muPhin;

        f_1_d[idx] = Af;  f_2_d[idx] = Af;
        f_3_d[idx] = Af;  f_4_d[idx] = Af;
        f_5_d[idx] = Af;  f_6_d[idx] = Af;
    }
}

