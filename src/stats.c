//-------------------------------------------------------------------------------
// Function : stats
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Intermediate simulation results, mostly for sanity checks. Includes average
// and maximum velocity in the domain as well as mass conservation. 
// temp[0] = volume; temp[1] = ux_av; temp[2] = uy_av; temp[3] = uz_av;
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

int stats( float *phi, float *ux, float *uy, float *uz )
{
    FILE  *fp;
    int   i, idx, j, k;
    float Umax, UmaxLocal, ux_av, uy_av, uz_av;
    float invVol, Ref, Usq, Vef;
    float temp[4], tempLocal[4];

    temp[0] = temp[1] = temp[2] = temp[3] = UmaxLocal = 0.f;
    tempLocal[0] = tempLocal[1] = tempLocal[2] = tempLocal[3] = 0.f;

    for( i = 1; i < NX_h-1; i++ ){
        for( j = 1; j < NY_h-1; j++ ){
            for( k = 1; k < NZ_h-1; k++ ){
                idx = gridId_h( i, j, k );

                Usq = ux[idx]*ux[idx] + uy[idx]*uy[idx] + uz[idx]*uz[idx];
                if( Usq > UmaxLocal ) UmaxLocal = Usq;

                if( phi[idx] >= 0.f ){
                    tempLocal[0] += 1.f;
                    tempLocal[1] += ux[idx];
                    tempLocal[2] += uy[idx];
                    tempLocal[3] += uz[idx];
                }
            }
        }
    }

    MPI_Reduce( &UmaxLocal, &Umax, 1, MPI_FLOAT, MPI_MAX, master, MPI_COMM_WORLD );
    MPI_Reduce( tempLocal,  temp,  4, MPI_FLOAT, MPI_SUM, master, MPI_COMM_WORLD );

    if( proc == master ){

        Umax   = sqrtf( Umax );
        invVol = 1.f/temp[0];
        if( step == 0 ){
            if( RELAX == 0 ){
                invInitVol = invVol;
                fp = fopen( "stats.out", "w" );
            }
            else{
                invRelaxVol = invVol; 
                fp = fopen( "relax_stats.out", "w" );
            }
            fprintf( fp, "#step\t\tux\t\tuy\t\tuz\t\tUmax\t\tVef\t\tRef\n" );
            fclose( fp );
        }

        ux_av = temp[1]*invVol;
        uy_av = temp[2]*invVol;
        uz_av = temp[3]*invVol;

        Ref = pow( temp[0]*invPi*0.75f, 1.f/3.f);

        if( RELAX == 0 ){
            Vef = temp[0]*invInitVol;
            fp = fopen( "stats.out", "a" );
        }
        else{
            Vef = temp[0]*invRelaxVol;
            fp = fopen( "relax_stats.out", "a" );
        }
        fprintf( fp, "%8d\t%e\t%e\t%e\t%e\t%e\t%e\n", step, ux_av, uy_av, uz_av, Umax, Vef, Ref );
        fclose( fp );
    }

    return 0;
}
