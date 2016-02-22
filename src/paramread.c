//-------------------------------------------------------------------------------
// Function : paramRead
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Read input parameters from ascii file "properties.in". 
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


int paramRead()
{
    FILE  *fp;
    char  tmp[32];
    int   i;
    int   buf_int[10];
    float invTauRho, invTauRhoHalf;
    float buf_flt[8], buf_bbl[80];

    // Get simulation data from primary configuration file
    if( proc == master ){
        fp = fopen( "properties.in", "r" );

        fscanf( fp, "%s %s %s %s", tmp, tmp, tmp, tmp );
        fscanf( fp, "%d %d %d %d", &maxStep, &relaxStep, &save, &stat );

        fscanf( fp, "%s %s %s", tmp, tmp, tmp );
        fscanf( fp, "%d %d %d", &xmax, &ymax, &zmax );

        fscanf( fp, "%s %s", tmp, tmp );
        fscanf( fp, "%d %d", &BLOCK_SIZE_X, &BLOCK_SIZE_Y );

        fscanf( fp, "%s %s", tmp, tmp );
        fscanf( fp, "%f %f", &rhoL, &rhoH );
         
        fscanf( fp, "%s %s", tmp, tmp );
        fscanf( fp, "%f %f", &tauRho, &tauPhi );

        fscanf( fp, "%s %s %s", tmp, tmp, tmp );
        fscanf( fp, "%f %f %f", &width, &sigma, &Gamma );

        fscanf( fp, "%s", tmp );
        fscanf( fp, "%f", &Eo );

        fclose( fp );

        // Get discrete phase data from secondary configuration file
        fp = fopen( "discrete.in", "r" );
        
        fscanf( fp, "%s", tmp );
        fscanf( fp, "%d", &nBubbles );

        fscanf( fp, "%s %s %s %s", tmp, tmp, tmp, tmp );
        for( i = 0; i < nBubbles; i++){
            fscanf( fp, "%f %f", &bubbles[i][0], &bubbles[i][1] ); // xc(i), yc(i)
            fscanf( fp, "%f %f", &bubbles[i][2], &bubbles[i][3] ); // zc(i), Rc(i)
        }
        fclose( fp );

        // Store input data in temporary arrays for MPI exchange
        buf_int[0] = maxStep;
        buf_int[1] = relaxStep;
        buf_int[2] = save;
        buf_int[3] = stat;
        buf_int[4] = xmax;
        buf_int[5] = ymax;
        buf_int[6] = zmax;
        buf_int[7] = BLOCK_SIZE_X;
        buf_int[8] = BLOCK_SIZE_Y;
        buf_int[9] = nBubbles;
        
        buf_flt[0] = rhoL;
        buf_flt[1] = rhoH;
        buf_flt[2] = tauRho;
        buf_flt[3] = tauPhi;
        buf_flt[4] = width;
        buf_flt[5] = sigma;
        buf_flt[6] = Gamma;
        buf_flt[7] = Eo;

        for( i = 0; i < nBubbles; i++ ){
            buf_bbl[4*i  ] = bubbles[i][0];
            buf_bbl[4*i+1] = bubbles[i][1];
            buf_bbl[4*i+2] = bubbles[i][2];
            buf_bbl[4*i+3] = bubbles[i][3];
        }
    }

    // Broadcast input data from processor 0
    MPI_Bcast( buf_int, 10, MPI_INT,   master, MPI_COMM_WORLD );
    MPI_Bcast( buf_flt, 8,  MPI_FLOAT, master, MPI_COMM_WORLD );
    MPI_Bcast( buf_bbl, 80, MPI_FLOAT, master, MPI_COMM_WORLD );

    // Copy input data from temporary arrays to local variables
    if( proc != master ){
        maxStep      = buf_int[0];
        relaxStep    = buf_int[1];
        save         = buf_int[2];
        stat         = buf_int[3];
        xmax         = buf_int[4];
        ymax         = buf_int[5];
        zmax         = buf_int[6];
        BLOCK_SIZE_X = buf_int[7];
        BLOCK_SIZE_Y = buf_int[8];
        nBubbles     = buf_int[9];
        
        rhoL   = buf_flt[0];
        rhoH   = buf_flt[1];
        tauRho = buf_flt[2];
        tauPhi = buf_flt[3];
        width  = buf_flt[4];
        sigma  = buf_flt[5];
        Gamma  = buf_flt[6];
        Eo     = buf_flt[7];

        for( i = 0; i < nBubbles; i++ ){
            bubbles[i][0] = buf_bbl[4*i  ];
            bubbles[i][1] = buf_bbl[4*i+1];
            bubbles[i][2] = buf_bbl[4*i+2];
            bubbles[i][3] = buf_bbl[4*i+3];
        }
    }


    // Fluid properties
    phiStar       = 0.5f*( rhoH - rhoL );
    phiStarSq     = phiStar*phiStar;
    invTauRho     = 1.f/tauRho;
    invTauRhoOne  = 1.f - invTauRho;
    invTauRhoHalf = 1.f - 0.5f*invTauRho;

    eta          = 1.f/( tauPhi + 0.5f );
    eta2         = 1.f - eta;
    invEta2      = 0.5f/( 1.f - eta );
    invTauPhi    = 1.f/tauPhi;
    invTauPhiOne = 1.f - invTauPhi;

    // Chemical Potential Stuff
    alpha  = 0.75f*sigma/( width*phiStarSq*phiStarSq );
    alpha4 = alpha*4.f;
    kappa  = ( width*phiStar )*( width*phiStar )*alpha*0.5f;

    // Rc = bubbles(1,4) Use the first bubble as the largest in the system
    gravity = 0.25f*Eo*sigma/( ( rhoH - rhoL )*bubbles[0][3]*bubbles[0][3] );
    grav    = 2.f*phiStar*gravity;

    // Modified LBM parameters
    Wn0  = invTauRho*W0;
    Wn1  = invTauRho*W1;
    Wn2  = invTauRho*W2;

    WnC0 = invTauRhoHalf*WC0;
    WnC1 = invTauRhoHalf*WC1;
    WnC2 = invTauRhoHalf*WC2;


    return 0;
}

