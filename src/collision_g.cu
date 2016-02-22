//-------------------------------------------------------------------------------
// Function : collision_g
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Collision step for link directions (0-18) of the momentum distribution.
// Includes gravitational force (downward along the z axis). 
// This function requires 28 arguments.
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

__global__ void collision_g( int *nb_east_d,  int *nb_west_d, 
                             int *nb_north_d, int *nb_south_d,
                             float *phi_d,  float *rho_d,  float *ux_d,
                             float *uy_d,   float *uz_d,   float *g_0_d,
                             float *g_1_d,  float *g_2_d,  float *g_3_d,
                             float *g_4_d,  float *g_5_d,  float *g_6_d,
                             float *g_7_d,  float *g_8_d,  float *g_9_d,
                             float *g_10_d, float *g_11_d, float *g_12_d,
                             float *g_13_d, float *g_14_d, float *g_15_d,
                             float *g_16_d, float *g_17_d, float *g_18_d )
{
    int   i, idx, idx2, ie, iw, j, jn, js, k, kt, kb;
    float phin, muPhi, rhon, eq1, eq2;
    float temp, lapPhi, Fx, Fy, Fz;
    float uxn, uyn, uzn, UF;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    // Main collision loop
    for( k = 1; k < NZ-1; k++ ){

        // Define some local values
        idx  = gridId( i, j, k );
        idx2 = gridId( i, j, k ) + dcol;
        phin = phi_d[idx];
        rhon = rho_d[idx];

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
        uxn = ux_d[idx] + 0.5f * Fx / rhon;
        uyn = uy_d[idx] + 0.5f * Fy / rhon;
        uzn = uz_d[idx] + 0.5f * Fz / rhon;

        // Collision for momentum distribution function
        // The factor invTauRho for geq is embedded in Egn
        //temp = rhon + K1*( 3.f*phin*muPhi 
        //     - 1.5f*rhon*( uxn*uxn + uyn*uyn + uzn*uzn ) );
        temp = K1*( 3.f*phin*muPhi + rhon*( 1.f
             - 1.5f*( uxn*uxn + uyn*uyn + uzn*uzn ) ) ); 
        UF   = uxn*Fx + uyn*Fy + uzn*Fz;

        // Rescale velocity to avoid unnecessary product operations
        uxn = uxn*invCs_sq;
        uyn = uyn*invCs_sq;
        uzn = uzn*invCs_sq;

        // The equilibrium g value and the force are bundled into gFs
        // DIRECTION 0
        // The 1/6.f factor is because we have rescaled uxn,uyn,uzn by invCs_Sq
        g_0_d[idx2] = invTauRhoOne_d*g_0_d[idx] 
                    + K0*( rhon - 6.f*phin*muPhi - rhon*( uxn*uxn + uyn*uyn
                    + uzn*uzn )/6.f ) - KC0*UF;

        // DIRECTIONS 1 & 2
        eq1 = temp + K1*rhon*uxn*uxn*0.5f + KC1*( uxn*Fx - UF );
        eq2 = uxn*( K1*rhon + KC1*Fx );
        g_1_d[idx2] = invTauRhoOne_d*g_1_d[idx] + eq1 + eq2;
        g_2_d[idx2] = invTauRhoOne_d*g_2_d[idx] + eq1 - eq2;

        // DIRECTIONS 3 & 4
        eq1 = temp + K1*rhon*uyn*0.5f*uyn + KC1*( uyn*Fy - UF );
        eq2 = uyn*( K1*rhon + KC1*Fy );
        g_3_d[idx2] = invTauRhoOne_d*g_3_d[idx] + eq1 + eq2;
        g_4_d[idx2] = invTauRhoOne_d*g_4_d[idx] + eq1 - eq2;

        // DIRECTIONS 5 & 6
        eq1 = temp + K1*rhon*uzn*0.5f*uzn + KC1*( uzn*Fz - UF );
        eq2 = uzn*( K1*rhon + KC1*Fz );
        g_5_d[idx2] = invTauRhoOne_d*g_5_d[idx] + eq1 + eq2;
        g_6_d[idx2] = invTauRhoOne_d*g_6_d[idx] + eq1 - eq2;


        // Notice we reset "temp" to use the correct K2 constant
        // The 1/6.f factor is because we have rescaled uxn,uyn,uzn by invCs_Sq
        temp = K2*( 3.f*phin*muPhi + rhon*( 1.f 
             - ( uxn*uxn + uyn*uyn + uzn*uzn )/6.f ) );

        // DIRECTION 7 & 8
        eq1 = temp + K2*rhon*( uxn + uyn )*0.5f*( uxn + uyn )
            + KC2*( ( uxn + uyn )*( Fx + Fy ) - UF );
        eq2 = K2*rhon*( uxn + uyn ) + KC2*( Fx + Fy );

        g_7_d[idx2] = invTauRhoOne_d*g_7_d[idx] + eq1 + eq2;
        g_8_d[idx2] = invTauRhoOne_d*g_8_d[idx] + eq1 - eq2;

        // DIRECTIONS 9 & 10
        eq1 = temp + K2*rhon*( uxn - uyn )*0.5f*( uxn - uyn )
            + KC2*( ( uxn - uyn )*( Fx - Fy ) - UF );
        eq2 = K2*rhon*( uxn - uyn ) + KC2*( Fx - Fy );
        g_9_d[idx2] = invTauRhoOne_d*g_9_d[idx] + eq1 + eq2;
        g_10_d[idx2] = invTauRhoOne_d*g_10_d[idx] + eq1 - eq2;

        // DIRECTIONS 11 & 12
        eq1 = temp + K2*rhon*( uxn + uzn )*0.5f*( uxn + uzn )
            + KC2*( ( uxn + uzn )*( Fx + Fz ) - UF );
        eq2 = K2*rhon*( uxn + uzn ) + KC2*( Fx + Fz );
        g_11_d[idx2] = invTauRhoOne_d*g_11_d[idx] + eq1 + eq2;
        g_12_d[idx2] = invTauRhoOne_d*g_12_d[idx] + eq1 - eq2;

        // DIRECTIONS 13 & 14
        eq1 = temp + K2*rhon*( uxn - uzn )*0.5f*( uxn - uzn )
            + KC2*( ( uxn - uzn )*( Fx - Fz ) - UF );
        eq2 = K2*rhon*( uxn - uzn ) + KC2*( Fx - Fz );
        g_13_d[idx2] = invTauRhoOne_d*g_13_d[idx] + eq1 + eq2;
        g_14_d[idx2] = invTauRhoOne_d*g_14_d[idx] + eq1 - eq2;

        // DIRECTIONS 15 & 16
        eq1 = temp + K2*rhon*( uyn + uzn )*0.5f*( uyn + uzn )
            + KC2*( ( uyn + uzn )*( Fy + Fz ) - UF );
        eq2 = K2*rhon*( uyn + uzn ) + KC2*( Fy + Fz );
        g_15_d[idx2] = invTauRhoOne_d*g_15_d[idx] + eq1 + eq2;
        g_16_d[idx2] = invTauRhoOne_d*g_16_d[idx] + eq1 - eq2;

        // DIRECTIONS 17 & 18
        eq1 = temp + K2*rhon*( uyn - uzn )*0.5f*( uyn - uzn )
            + KC2*( ( uyn - uzn )*( Fy - Fz ) - UF );
        eq2 = K2*rhon*( uyn - uzn ) + KC2*( Fy - Fz );
        g_17_d[idx2] = invTauRhoOne_d*g_17_d[idx] + eq1 + eq2;
        g_18_d[idx2] = invTauRhoOne_d*g_18_d[idx] + eq1 - eq2;

    }

}

