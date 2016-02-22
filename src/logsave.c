//-------------------------------------------------------------------------------
// Function : logSave
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Save relevant data at the end of the simulation run to file 'final.out'.
//     tempLocal[0] : volume
//     tempLocal[1] : ux
//     tempLocal[2] : uy
//     tempLocal[3] : uz
//     tempLocal[4] : pressureIn
//     tempLocal[5] : pressureOut
// This function requires 11 arguments.
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

int logSave( char *devName, float devMem, int *nb_east, int *nb_west,
             int *nb_north, int *nb_south, float *phi, float *rho, 
             float *ux, float *uy, float *uz )
{
    FILE  *fp, *pipe;
    int   i, idx, ie, iw, j, jn, js, k, kt, kb;
    float Pin, Pout, Ro, Pdif, Perr, Ref, Vef;
    float auxMem, distMem, hydroMem, mpiMem, nbMem, gpuMem, cpuMem;
    float Umax, UmaxLocal, Usq, phin, phin2, pressure;
    float gradPhiX, gradPhiY, gradPhiZ, gradPhiSq, lapPhi;
    float p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, 
          p14, p15, p16, p17, p18;
    float volumeIn, volumeOut, lbstep, mlups;
    float temp[6], tempLocal[6];
    char  dateStamp[11], timeStamp[6];

//========================== Initialization ====================================
    Ro = bubbles[0][3];

    volumeIn    = 0.f; volumeOut   = 0.f;

    Umax    = 0.f; UmaxLocal    = 0.f;
    temp[0] = 0.f; tempLocal[0] = 0.f;
    temp[1] = 0.f; tempLocal[1] = 0.f;
    temp[2] = 0.f; tempLocal[2] = 0.f;
    temp[3] = 0.f; tempLocal[3] = 0.f;
    temp[4] = 0.f; tempLocal[4] = 0.f;
    temp[5] = 0.f; tempLocal[5] = 0.f;

//================== Final configuration statistics ============================
    for( i = 1; i < NX_h-1; i++ ){
        for( j = 1; j < NY_h-1; j++ ){
            for( k = 1; k < NZ_h-1; k++ ){

                idx = gridId_h( i, j, k );

//================= Maximum velocity in simulation domain ======================
                Usq = ux[idx]*ux[idx] + uy[idx]*uy[idx] + uz[idx]*uz[idx];
                if( Usq > UmaxLocal ) UmaxLocal = Usq;

//================= Average bubble velocity and volume conservation ============
                if( phi[idx] >= 0.f ){
                    tempLocal[0] += 1.f;
                    tempLocal[1] += ux[idx];
                    tempLocal[2] += uy[idx];
                    tempLocal[3] += uz[idx];
                }

//================= Pressure inside and outside bubble =========================
// Notice that this calculation is only relevant during the relaxation period, 
// as Laplace's equation applies to a static case only.
//==============================================================================
                phin  = phi[idx];
                phin2 = phin*phin; 
                ie    = nb_east[idx];
                iw    = nb_west[idx];
                jn    = nb_north[idx];
                js    = nb_south[idx];
                kt    = k + 1;
                kb    = k - 1;

                // Order parameter values along each of the 19 directions
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

                gradPhiSq = gradPhiX * gradPhiX
                          + gradPhiY * gradPhiY
                          + gradPhiZ * gradPhiZ;

                // Calculate the pressure
                pressure = alpha*( phin2*( 3.f*phin2 - 2.f*phiStarSq ) 
                         - phiStarSq*phiStarSq ) - kappa*( phin*lapPhi 
                         - 0.5f*gradPhiSq ) + Cs_sq*rho[idx];

                if( phi[idx] >= 0.f ){
                    tempLocal[4] += pressure;
                }else{
                    tempLocal[5] += pressure;
                }

            }
        }
    }

    MPI_Reduce( &UmaxLocal, &Umax, 1, MPI_FLOAT, MPI_MAX, master, MPI_COMM_WORLD );
    MPI_Reduce( tempLocal,  temp,  6, MPI_FLOAT, MPI_SUM, master, MPI_COMM_WORLD );


//============= Use task 0 to format and output information ====================
    if( proc == master ){

        // Calculate compliance with Laplace's Law ( single bubble, static case )
        volumeIn  = temp[0]; 
        volumeOut = NX_h*NY_h*( NZ_h - 2 ) - temp[0];
        Pin  = temp[4] / volumeIn;
        Pout = temp[5] / volumeOut;
        Pdif = Pin - Pout;
        Perr = 0.5f*( 2.f*sigma/Ro - Pdif )*Ro/sigma;

        // Calculate phase conservation
        Ref = pow( volumeIn*invPi*0.75f, 1.f/3.f );
        if( RELAX == 0 ) Vef = volumeIn*invInitVol;
        if( RELAX == 1 ) Vef = volumeIn*invRelaxVol;

       // Estimate memory usage (bytes)
        mpiMem   = 4.f*sizeof(float)*bufSize             // phi and f exchange
                 + 4.f*6.f*sizeof(float)*bufSize;        // f and g exchange
        nbMem    = 4.f*sizeof(int)*gridSize;             // nb_east, nb_west, ...
        hydroMem = 5.f*sizeof(float)*gridSize;           // phi, rho, ux, uy, uz
        distMem  = 2.f*( 19.f*sizeof(float)*gridSize     // g distribution function 
                 +        7.f*sizeof(float)*gridSize );  // f distribution function
        auxMem   = nprocs * ( nbMem + hydroMem + mpiMem );
        gpuMem   = nprocs * ( nbMem + hydroMem + mpiMem + distMem );
        cpuMem   = nprocs * ( nbMem + hydroMem + mpiMem );

        // Write data to file
        if( RELAX == -1 ){
            fp = fopen( "runlog.out", "w" );

            fprintf( fp, "***\n" );
            fprintf( fp, "*** Multiphase Zheng-Shu-Chew LBM 3D Simulation  \n" );
            fprintf( fp, "*** CUDA MPI Implementation Version 13.0-devel \n" );
            fprintf( fp, "***\n" );

            fprintf( fp, "*** Program running on : %s\n", devName );
            if( devMem >= GB ){
                fprintf( fp, "*** Total GPU Memory   : %.4f GB\n", devMem / GB);
            }else{
                fprintf( fp, "*** Total GPU Memory   : %.4f MB\n", devMem / MB);
            }
            fprintf( fp, "***\n" );

            pipe = popen( "date +%R", "r" );
            fgets( timeStamp, 6, pipe ); pclose( pipe );
            pipe = popen( "date +%F", "r" );
            fgets( dateStamp, 11, pipe ); pclose( pipe );
            fprintf( fp, "*** Simulation started at %s on %s\n", timeStamp, dateStamp );
            fprintf( fp, "***\n\n" );

            fprintf( fp, "INPUT PARAMETERS\n" );
            fprintf( fp, "Evolution Steps    = %d\n", maxStep   );
            fprintf( fp, "Relaxation Steps   = %d\n", relaxStep );
            fprintf( fp, "Domain Size in X   = %d\n", xmax );
            fprintf( fp, "Domain Size in Y   = %d\n", ymax );
            fprintf( fp, "Domain Size in Z   = %d\n", zmax );
            fprintf( fp, "Interface Width    = %6.3f\n", width );
            fprintf( fp, "Interface Tension  = %6.3f\n", sigma );
            fprintf( fp, "Interface Mobility = %6.3f\n", Gamma );
            fprintf( fp, "Low Fluid Density  = %6.3f\n", rhoL );
            fprintf( fp, "High Fluid Density = %6.3f\n", rhoH );
            fprintf( fp, "Tau (Density)      = %6.3f\n", tauRho );
            fprintf( fp, "Tau (Phase)        = %6.3f\n", tauPhi );
            fprintf( fp, "\n" );

            fprintf( fp, "CUDA CONFIGURATION\n" );
            fprintf( fp, "GPUs used    = %d\n", nprocs );
            fprintf( fp, "BLOCK_SIZE_X = %d\n", BLOCK_SIZE_X );
            fprintf( fp, "BLOCK_SIZE_Y = %d\n", BLOCK_SIZE_Y );
            fprintf( fp, "\n" );

            if( gpuMem < GB ){
                fprintf( fp, "MEMORY USAGE (Mb)\n" );
                fprintf( fp, "Distributions         = %4.2f\n", distMem / MB );
                fprintf( fp, "Auxiliary Arrays      = %4.2f\n", auxMem / MB );
                fprintf( fp, "Total CPU Memory Used = %4.2f\n", cpuMem / MB );
                fprintf( fp, "Total GPU Memory Used = %4.2f\n", gpuMem / MB );
            }else{
                fprintf( fp, "MEMORY USAGE (Gb)\n" );
                fprintf( fp, "Distributions         = %4.2f\n", distMem / GB );
                fprintf( fp, "Auxiliary Arrays      = %4.2f\n", auxMem / GB );
                fprintf( fp, "Total CPU Memory Used = %4.2f\n", cpuMem / GB );
                fprintf( fp, "Total GPU Memory Used = %4.2f\n", gpuMem / GB );
            }
            fprintf( fp, "\n" );

            fclose( fp );
        }else if( RELAX == 1 ){
            fp = fopen( "runlog.out", "a" );

            fprintf( fp, "RELAXATION RESULTS\n" );
            fprintf( fp, "Effective Radius   = %e\n", Ref );
            fprintf( fp, "Phase Conservation = %e\n", Vef );
            fprintf( fp, "(Pin - Pout)       = %e\n", Pdif );
            fprintf( fp, "Laplace Error      = %e\n", Perr );
            fprintf( fp, "Parasitic Velocity = %e\n", Umax );
            fprintf( fp, "\n" );

            fclose( fp );
        }else{
            fp = fopen( "runlog.out", "a" );

            fprintf( fp, "OUTPUT RESULTS\n" );
            fprintf( fp, "Effective Radius   = %e\n", Ref );
            fprintf( fp, "Phase Conservation = %e\n", Vef );
            fprintf( fp, "Maximum Velocity   = %e\n", Umax );
            fprintf( fp, "\n" );

            lbstep = mainTime * 0.001f / ( (float)maxStep );
            mlups  = (float)nprocs * (float)maxStep * (float)gridSize * 0.000001f / ( mainTime * 0.001f );
            fprintf( fp, "TIMING INFORMATION (seconds)\n" );
            fprintf( fp, "Setup Time      = %9.6f\n", setupTime*0.001f );
            fprintf( fp, "Relaxation Time = %9.6f\n", relaxTime*0.001f );
            fprintf( fp, "Main Loop Time  = %9.6f\n", mainTime*0.001f );
            fprintf( fp, "LBM step Time   = %9.6f\n", lbstep );
            fprintf( fp, "MLUPS           = %9.6f\n", mlups );
            fprintf( fp, "\n" );

            pipe = popen( "date +%R", "r" );
            fgets( timeStamp, 6, pipe ); pclose( pipe );
            pipe = popen( "date +%F", "r" );
            fgets( dateStamp, 11, pipe ); pclose( pipe );
            fprintf( fp, "***\n" );
            fprintf( fp, "*** Simulation completed at %s on %s\n", timeStamp, dateStamp );
            fprintf( fp, "***\n" );
            
            fclose( fp );
        }
    }

    return 0;
}
