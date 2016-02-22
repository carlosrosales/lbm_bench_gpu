//-------------------------------------------------------------------------------
// Function : vtkSave
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Intermediate simulation results, mostly for sanity checks. Includes average
// and maximum velocity in the domain as well as mass conservation. 
//
// This function has 9 arguments.
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

int vtkSave( int *nb_east, int *nb_west, int *nb_north, int *nb_south,
             float *phi, float *rho, float *ux, float *uy, float *uz )
{
    FILE  *fp;
    char  vtkfile[64];
    int   i, idx, ie, iw, j, jn, js, k, kt, kb;
    float phin, phin2, pressure;
    float gradPhiX, gradPhiY, gradPhiZ, gradPhiSq, lapPhi;
    float p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13,
          p14, p15, p16, p17, p18;
   
    if( RELAX == 0 ){ sprintf( vtkfile, "data_%02d_%07d.vtk", vproc, step ); }
    else{             sprintf( vtkfile, "relax_data_%02d_%07d.vtk", vproc, step ); }
    fp = fopen( vtkfile, "w" );

    fprintf( fp, "# vtk DataFile Version 2.0\n" );
    fprintf( fp, "CUDA-MPI ZSC v13.0-devel\n" );
    fprintf( fp, "ASCII\n" );
    fprintf( fp, "\n" );
    fprintf( fp, "DATASET STRUCTURED_POINTS\n" );
    fprintf( fp, "DIMENSIONS %d %d %d\n", NX_h, NY_h, NZ_h-2 );
    fprintf( fp, "ORIGIN %d %d %d\n", 1, 1, zl );
    fprintf( fp, "SPACING %d %d %d\n", 1, 1, 1 );
    fprintf( fp, "\n" );
    fprintf( fp, "POINT_DATA %d\n", NX_h*NY_h*(NZ_h-2) );
    fprintf( fp, "\n" );
    fprintf( fp, "SCALARS Phi float\n" );
    fprintf( fp, "LOOKUP_TABLE default\n" );
    for( k = 1; k < NZ_h-1; k++ )
        for( j = 0; j < NY_h; j++ )
            for( i = 0; i < NX_h; i++ )
                fprintf( fp, "%f\n", phi[ gridId_h( i, j, k ) ] );

    fprintf( fp, "\n" );
    fprintf( fp, "SCALARS Pressure float\n" );
    fprintf( fp, "LOOKUP_TABLE default\n" );
    for( k = 1; k < NZ_h-1; k++ ){
        for( j = 0; j < NY_h; j++ ){
            for( i = 0; i < NX_h; i++ ){

                idx = gridId_h( i, j, k );

                phin  = phi[idx];
                phin2 = phin*phin;                

                // Identify neighbours
                ie = nb_east[idx];
                iw = nb_west[idx];
                jn = nb_north[idx];
                js = nb_south[idx];
                kt = k + 1;
                kb = k - 1;

                // Nodal values of the order parameter
                p00 = phi[ gridId_h(i ,j ,k ) ];
                p01 = phi[ gridId_h(ie,j ,k ) ];
                p02 = phi[ gridId_h(iw,j ,k ) ];
                p03 = phi[ gridId_h(i ,jn,k ) ];
                p04 = phi[ gridId_h(i ,js,k ) ];
                p05 = phi[ gridId_h(i ,j ,kt) ];
                p06 = phi[ gridId_h(i ,j ,kb) ];
                p07 = phi[ gridId_h(ie,jn,k ) ];
                p08 = phi[ gridId_h(iw,js,k ) ];
                p09 = phi[ gridId_h(ie,js,k ) ];
                p10 = phi[ gridId_h(iw,jn,k ) ];
                p11 = phi[ gridId_h(ie,j ,kt) ];
                p12 = phi[ gridId_h(iw,j ,kb) ];
                p13 = phi[ gridId_h(ie,j ,kb) ];
                p14 = phi[ gridId_h(iw,j ,kt) ];
                p15 = phi[ gridId_h(i ,jn,kt) ];
                p16 = phi[ gridId_h(i ,js,kb) ]; 
                p17 = phi[ gridId_h(i ,jn,kb) ];
                p18 = phi[ gridId_h(i ,js,kt) ];

                // Laplacian of the order parameter
                lapPhi = ( p07 + p08 + p09 + p10 + p11 + p12 + p13 + p14
                       +   p15 + p16 + p17 + p18 + 2.f*( p01 + p02 + p03
                       +   p04 + p05 + p06 - 12.f*p00 ) )*inv6;

                // Components of the order parameter gradient
                gradPhiX = ( 2.f*( p01 - p02 ) + p07 - p08 + p09 - p10
                         +         p11 - p12   + p13 - p14 )*inv12;

                gradPhiY = ( 2.f*( p03 - p04 ) + p07 - p08 + p10 - p09
                         +         p15 - p16   + p17 - p18 )*inv12;

                gradPhiZ = ( 2.f*( p05 - p06 ) + p11 - p12 + p14 - p13
                         +         p15 - p16   + p18 - p17 )*inv12;

                gradPhiSq = gradPhiX*gradPhiX 
                          + gradPhiY*gradPhiY
                          + gradPhiZ*gradPhiZ;

                // Calculate the pressure
                pressure = alpha*( phin2*( 3.f*phin2 - 2.f*phiStarSq ) 
                         - phiStarSq*phiStarSq ) - kappa*( phin*lapPhi 
                         - 0.5f*gradPhiSq ) + Cs_sq*rho[idx];

                fprintf( fp, "%f\n", pressure );
            }
        }
    }

    fprintf( fp, "\n" );
    fprintf( fp, "Vectors Velocity float\n" );
    for( k = 1; k < NZ_h-1; k++ ){
        for( j = 0; j < NY_h; j++ ){
            for( i = 0; i < NX_h; i++ ){
                idx = gridId_h( i, j, k );
                fprintf( fp, "%f %f %f\n", ux[idx], uy[idx], uz[idx] );
            }
        }
    }
    fclose( fp );

    return 0;
}

