//-------------------------------------------------------------------------------
// Function : collision_f
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Collision step. Includes gravitational force (downward along the z axis). 
// This function requires 12 arguments.
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

__global__ void collision_f( int *nb_east_d,  int *nb_west_d, 
                             int *nb_north_d, int *nb_south_d,
                             float *phi_d,  float *rho_d, float *ux_d,  
                             float *uy_d,   float *uz_d,  float *f_0_d,
                             float *f_1_d,  float *f_2_d, float *f_3_d, 
                             float *f_4_d,  float *f_5_d, float *f_6_d )
{
    int   i, idx, idx2, ie, iw, j, jn, js, k, kt, kb;
    float phin, muPhi, invRho;
    float lapPhi, Fx, Fy, Fz;
    float Af, Cf, uxn, uyn, uzn;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    // Main collision loop
    for( k = 1; k < NZ-1; k++ ){

        // Define some local values
        idx    = gridId( i, j, k );
        idx2   = gridId( i, j, k ) + dcol;
        phin   = phi_d[idx];
        invRho = 1.f / rho_d[idx];

        // Near neighbors
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

        // Force calculation ( +grav only if phi > 0 )
        // Fx = muPhi*gradPhiX
        // Fy = muPhi*gradPhiY
        // Fz = muPhi*gradPhiZ + grav
        muPhi = alpha4_d*phin*( phin*phin - phiStarSq_d ) - kappa_d*lapPhi;

        Fx = muPhi*( 2.f*( phi_d[ gridId(ie,j ,k ) ] - phi_d[ gridId(iw,j ,k ) ] ) 
           + phi_d[ gridId(ie,jn,k ) ] - phi_d[ gridId(iw,js,k ) ] 
           + phi_d[ gridId(ie,js,k ) ] - phi_d[ gridId(iw,jn,k ) ] 
           + phi_d[ gridId(ie,j ,kt) ] - phi_d[ gridId(iw,j ,kb) ]
           + phi_d[ gridId(ie,j ,kb) ] - phi_d[ gridId(iw,j ,kt) ] )*inv12;

        Fy = muPhi*( 2.f*( phi_d[ gridId(i ,jn,k ) ] - phi_d[ gridId(i ,js,k ) ] ) 
           + phi_d[ gridId(ie,jn,k ) ] - phi_d[ gridId(iw,js,k ) ] 
           + phi_d[ gridId(iw,jn,k ) ] - phi_d[ gridId(ie,js,k ) ]
           + phi_d[ gridId(i ,jn,kt) ] - phi_d[ gridId(i ,js,kb) ]
           + phi_d[ gridId(i ,jn,kb) ] - phi_d[ gridId(i ,js,kt) ] )*inv12;

        Fz = muPhi*( 2.f*( phi_d[ gridId(i ,j ,kt) ] - phi_d[ gridId(i ,j ,kb) ] )
           + phi_d[ gridId(ie,j ,kt) ] - phi_d[ gridId(iw,j ,kb) ]
           + phi_d[ gridId(iw,j ,kt) ] - phi_d[ gridId(ie,j ,kb) ]
           + phi_d[ gridId(i ,jn,kt) ] - phi_d[ gridId(i ,js,kb) ]
           + phi_d[ gridId(i ,js,kt) ] - phi_d[ gridId(i ,jn,kb) ] )*inv12;

         if( phin > 0.f ) Fz = Fz + grav_d;

        // Correct velocity calculation to include interfacial 
        // and gravitational forces and scale by invRho
        uxn = ux_d[idx] + 0.5f * Fx * invRho;
        uyn = uy_d[idx] + 0.5f * Fy * invRho;
        uzn = uz_d[idx] + 0.5f * Fz * invRho;

        // Collision for order parameter distribution function f/fcol
        Af = 0.5f * Gamma_d * invTauPhi_d * muPhi;
        Cf = invTauPhi_d * invEta2_d * phin;

        f_0_d[idx2] = invTauPhiOne_d * f_0_d[idx] - 6.0f*Af + invTauPhi_d*phin;
        f_1_d[idx2] = invTauPhiOne_d * f_1_d[idx] + Af + Cf * uxn;
        f_2_d[idx2] = invTauPhiOne_d * f_2_d[idx] + Af - Cf * uxn;
        f_3_d[idx2] = invTauPhiOne_d * f_3_d[idx] + Af + Cf * uyn;
        f_4_d[idx2] = invTauPhiOne_d * f_4_d[idx] + Af - Cf * uyn;
        f_5_d[idx2] = invTauPhiOne_d * f_5_d[idx] + Af + Cf * uzn;
        f_6_d[idx2] = invTauPhiOne_d * f_6_d[idx] + Af - Cf * uzn;

    }

}

