//-------------------------------------------------------------------------------
// Function : init
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Initialize all variables and arrays.
// Note that the z-direction domain decomposition makes it possible to avoid the
// use of ghost layers in X and Y
// This function requires 9 arguments.
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

int init( int *nb_east, int *nb_west, int *nb_north,  int *nb_south, 
          float *phi, float *rho, float *ux, float *uy, float *uz )
{
    int   i, idx, j, k, m;
    float dx, dy, dz, R, rc;

    // Initialize some constants
    step = 0;

    // Set near neighbors and order parameter
    for( i = 0; i < NX_h; i++){
        for( j = 0; j < NY_h; j++){
            for( k = 0; k < NZ_h; k++){

                idx      = gridId_h( i, j, k );
                phi[idx] = -phiStar;
                rho[idx] = 0.5f*( rhoH + rhoL );
                ux[idx]  = 0.f;
                uy[idx]  = 0.f;
                uz[idx]  = 0.f;

                for( m = 0; m < nBubbles; m++ ){
                    dx = (float)i - bubbles[m][0];
                    dy = (float)j - bubbles[m][1];
                    dz = (float)(k + zlg) - bubbles[m][2];
                    rc = bubbles[m][3];

                    R = sqrtf( dx*dx + dy*dy + dz*dz );
                    if( R <= ( rc + width ) )
                        phi[idx] = phiStar*tanhf( 2.f*( rc - R )/width );
                }

                // Assign neighbor values to arrays
                nb_east[idx]  = i + 1;
                nb_west[idx]  = i - 1;
                nb_north[idx] = j + 1;
                nb_south[idx] = j - 1;
            }
        }
    }
    
    // Fix values at boundaries
    for( j = 0; j < NY_h; j++ ){
        for( k = 0; k < NZ_h; k++ ){
            nb_west[ gridId_h( 0,      j, k ) ] = NX_h-1;
            nb_east[ gridId_h( NX_h-1, j, k ) ] = 0;
        }
    }
    for( i = 0; i < NX_h; i++ ){
        for( k = 0; k < NZ_h; k++ ){
            nb_south[ gridId_h( i, 0,      k ) ] = NY_h-1;
            nb_north[ gridId_h( i, NY_h-1, k ) ] = 0;
        }
    }


    return 0;
}
