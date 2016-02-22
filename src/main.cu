//-------------------------------------------------------------------------------
// Program   : lbm_bench_gpu
// Revision  : 1.0 (2016/02/22)
// Author    : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Driver for the hybrid MPI+OpenMP implementation of the Zheng-Shu-Chew
// multiphase LBM using D3Q7/D3Q19 discretization and periodic boundary
// conditions, including the gravitational force. For details:
//
// Journal of Computational Physics 218: 353-371, 2006.
//
// The average velocity, mass conservation factor, effective radius of the drop,
// pressure difference between the inside and the outside of the drop and the
// error with respect to the analytical value given by Laplace's equation are
// written to file "stats.out"
//
// The values of RELAX are used to control output:
// RELAX = -1 -> Setup Stage Completed
// RELAX =  0 -> Relaxation Stage Completed
// RELAX =  1 -> Main Calculation Stage Completed
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda.h>
#include "constants.h"

// CPU functions
#include "gridid.c"
#include "init.c"
#include "paramread.c"
#include "logsave.c"
#include "stats.c"
#include "mpiupdate.c"
#include "mpiupdate_f.c"
#include "mpiupdate_phi.c"
#include "vgrid.c"
#include "vtksave.c"

// GPU kernels
#include "init_f.cu"
#include "init_g.cu"
#include "update_phi.cu"
#include "update_rho.cu"
#include "update_velocity.cu"
#include "collision_f.cu"
#include "collision_g.cu"
#include "pack_mpi.cu"
#include "pack_mpi_f.cu"
#include "pack_mpi_phi.cu"
#include "stream_f.cu"
#include "stream_g.cu"
#include "unpack_mpi.cu"
#include "unpack_mpi_f.cu"
#include "unpack_mpi_phi.cu"

void fatalError( char *errorStr )
{
    if( proc == 0 ) fprintf( stderr, "ERROR: %s.\n", errorStr );
    MPI_Abort( MPI_COMM_WORLD, -1 );
}


int main( int argc, char **argv )
{
    int   device;
    float cpuMem, devMem, distMem, gpuMem, hydroMem, mpiMem, nbMem;
    char devName[32];

    // Initialize MPI environment
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // choose the GPU for execution
    device = proc % 2;
    cudaSetDevice(device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // Declare and create timing events
    cudaEvent_t start, stopSetup, stopRelax, stopMain;
    cudaEventCreate( &start );
    cudaEventCreate( &stopSetup  );
    cudaEventCreate( &stopRelax  );
    cudaEventCreate( &stopMain  );

    // Write some information about the card we are runing on
    // Assume all cards in the cluster are the same
    if( proc == 0 ){
        fprintf( stdout, "*** Multiphase Zheng-Shu-Chew LBM 3D Simulation \n" );
        fprintf( stdout, "*** CUDA MPI Implementation Version 12.0-devel  \n" );
        fprintf( stdout, "\n" );

        strncpy( devName, properties.name, 32 );
        devMem = properties.totalGlobalMem;
        fprintf( stdout, "*** Program running on : %s\n", devName );
        if( devMem >= GB ){
            fprintf( stdout, "*** Total GPU Memory   : %.4f GB\n\n", devMem / GB);
        }else{
            fprintf( stdout, "*** Total GPU Memory   : %.4f MB\n\n", devMem / MB);
        }
    }

//====================== ARRAY DECLARATIONS ====================================
    // Host array declarations. These arrays have dinensions NX*NY*NZ, and
    // can be traversed using the running index idx = j + NY*i
    int   *nb_east = 0, *nb_west = 0, *nb_north  = 0, *nb_south  = 0;

    float *top_snd  = 0, *bot_snd  = 0, *top_rcv  = 0, *bot_rcv  = 0;
    float *topF_snd = 0, *botF_snd = 0, *topF_rcv = 0, *botF_rcv = 0;

    float *phi = 0, *rho = 0, *ux = 0, *uy = 0, *uz = 0;

    // Device array declarations. These arrays have dinensions NX*NY*NZ, and
    // can be traversed using the running index idx = k + NZ*( j + NY*i )
    int   *nb_east_d = 0, *nb_west_d = 0, *nb_north_d  = 0, *nb_south_d  = 0;

    float *top_snd_d  = 0, *bot_snd_d  = 0, *top_rcv_d  = 0, *bot_rcv_d  = 0;
    float *topF_snd_d = 0, *botF_snd_d = 0, *topF_rcv_d = 0, *botF_rcv_d = 0;

    float *phi_d = 0, *rho_d = 0, *ux_d = 0, *uy_d = 0, *uz_d = 0;

    float *f_0_d = 0, *f_1_d = 0, *f_2_d = 0, *f_3_d = 0, *f_4_d = 0,
          *f_5_d = 0, *f_6_d = 0;

    float *g_0_d  = 0, *g_1_d  = 0, *g_2_d  = 0, *g_3_d  = 0, *g_4_d  = 0,
          *g_5_d  = 0, *g_6_d  = 0, *g_7_d  = 0, *g_8_d  = 0, *g_9_d  = 0,
          *g_10_d = 0, *g_11_d = 0, *g_12_d = 0, *g_13_d = 0, *g_14_d = 0,
          *g_15_d = 0, *g_16_d = 0, *g_17_d = 0, *g_18_d = 0;


//====================== READ INPUT PARAMETERS =================================
    cudaEventRecord( start, 0 );
    cudaEventSynchronize( start );
    RELAX = -1;
    paramRead();
    vgrid();

    NX_h  = xmax;
    NY_h  = ymax;
    NZ_h  = ( zug - zlg + 1 );
    NXY_h = NX_h * NY_h;
    bufSize  = NX_h*NY_h;
    gridSize = NX_h*NY_h*NZ_h;

    // Sanity check for domain size
    if( (NX_h%32) != 0 ) fatalError( "NX is not a multiple of 32");
    if( (NY_h%32) != 0 ) fatalError( "NY is not a multiple of 32");

    // Sanity check for partition configuration
    if( BLOCK_SIZE_X*BLOCK_SIZE_Y > 512 ) fatalError( "Too many blocks" );
    if( NX_h < BLOCK_SIZE_X ) fatalError( "BLOCK_SIZE_X is too large");
    if( NY_h < BLOCK_SIZE_Y ) fatalError( "BLOCK_SIZE_Y is too large");

    // Define the grid and the number of threads per block for the calculations
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 );
    dim3 dimgrid( NX_h/BLOCK_SIZE_X, NY_h/BLOCK_SIZE_Y, 1 );

//====================== HOST ARRAY MEMORY ALLOCATION ==========================
    nb_east  = (int *)malloc( gridSize*sizeof(int) );
    nb_west  = (int *)malloc( gridSize*sizeof(int) );
    nb_north = (int *)malloc( gridSize*sizeof(int) );
    nb_south = (int *)malloc( gridSize*sizeof(int) );

    top_snd = (float *)malloc( 6*bufSize*sizeof(float) );
    top_rcv = (float *)malloc( 6*bufSize*sizeof(float) );
    bot_snd = (float *)malloc( 6*bufSize*sizeof(float) );
    bot_rcv = (float *)malloc( 6*bufSize*sizeof(float) );

    topF_snd = (float *)malloc( bufSize*sizeof(float) );
    topF_rcv = (float *)malloc( bufSize*sizeof(float) );
    botF_snd = (float *)malloc( bufSize*sizeof(float) );
    botF_rcv = (float *)malloc( bufSize*sizeof(float) );

    phi = (float *)malloc( gridSize*sizeof(float) );
    rho = (float *)malloc( gridSize*sizeof(float) );

    ux = (float *)malloc( gridSize*sizeof(float) );
    uy = (float *)malloc( gridSize*sizeof(float) );
    uz = (float *)malloc( gridSize*sizeof(float) );

//====================== DEVICE ARRAY MEMORY ALLOCATION ========================
    cudaMalloc( (void **) &nb_east_d,  gridSize*sizeof(int) );
    cudaMalloc( (void **) &nb_west_d,  gridSize*sizeof(int) );
    cudaMalloc( (void **) &nb_north_d, gridSize*sizeof(int) );
    cudaMalloc( (void **) &nb_south_d, gridSize*sizeof(int) );

    cudaMalloc( (void **) &top_snd_d, 6*bufSize*sizeof(float) );
    cudaMalloc( (void **) &top_rcv_d, 6*bufSize*sizeof(float) );
    cudaMalloc( (void **) &bot_snd_d, 6*bufSize*sizeof(float) );
    cudaMalloc( (void **) &bot_rcv_d, 6*bufSize*sizeof(float) );

    cudaMalloc( (void **) &topF_snd_d, bufSize*sizeof(float) );
    cudaMalloc( (void **) &topF_rcv_d, bufSize*sizeof(float) );
    cudaMalloc( (void **) &botF_snd_d, bufSize*sizeof(float) );
    cudaMalloc( (void **) &botF_rcv_d, bufSize*sizeof(float) );

    cudaMalloc( (void **) &phi_d, gridSize*sizeof(float) );
    cudaMalloc( (void **) &rho_d, gridSize*sizeof(float) );

    cudaMalloc( (void **) &ux_d, gridSize*sizeof(float) );
    cudaMalloc( (void **) &uy_d, gridSize*sizeof(float) );
    cudaMalloc( (void **) &uz_d, gridSize*sizeof(float) );

    cudaMalloc( (void **) &f_0_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_1_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_2_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_3_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_4_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_5_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &f_6_d, 2*gridSize*sizeof(float) );

    cudaMalloc( (void **) &g_0_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_1_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_2_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_3_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_4_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_5_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_6_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_7_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_8_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_9_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_10_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_11_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_12_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_13_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_14_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_15_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_16_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_17_d, 2*gridSize*sizeof(float) );
    cudaMalloc( (void **) &g_18_d, 2*gridSize*sizeof(float) );

//====================== DEVICE ARRAY ALLOCATION TEST ==========================
    if( nb_north_d == 0 || nb_south_d == 0 || nb_east_d  == 0 || nb_west_d == 0 )
        fatalError( "Unable to allocate memory for neighbors on device" );

    if( top_snd_d == 0  || bot_snd_d == 0  || top_rcv_d == 0  || 
        bot_rcv_d == 0  || topF_snd_d == 0 || topF_rcv_d == 0 ||
        botF_snd_d == 0 || botF_rcv_d == 0 )
        fatalError( "Unable to allocate memory for MPI buffers on device" );

    if( phi_d == 0 || rho_d == 0 || ux_d == 0 || uy_d == 0 || uz_d == 0 )
        fatalError( "Unable to allocate memory for hydro variables on device" );

    if( f_0_d == 0 || f_1_d == 0 || f_2_d == 0 || f_3_d == 0 || 
        f_4_d == 0 || f_5_d == 0 || f_6_d == 0)
        fatalError( "Unable to allocate memory for f on device" );

    if( g_0_d  == 0 || g_1_d  == 0 || g_2_d  == 0 || g_3_d  == 0 || 
        g_4_d  == 0 || g_5_d  == 0 || g_6_d  == 0 || g_7_d  == 0 ||
        g_8_d  == 0 || g_9_d  == 0 || g_10_d == 0 || g_11_d == 0 ||
        g_12_d == 0 || g_13_d == 0 || g_14_d == 0 || g_15_d == 0 ||
        g_16_d == 0 || g_17_d == 0 || g_18_d == 0 )
        fatalError( "Unable to allocate memory for g on device" );


    nbMem    = 4.f*sizeof(int)*gridSize;
    mpiMem   = 4.f*sizeof(float)*bufSize*7.f;
    hydroMem = 5.f*sizeof(float)*gridSize;
    distMem  = 2.f*( 19.f*sizeof(float)*gridSize + 7.f*sizeof(float)*gridSize );
    gpuMem   = nbMem + mpiMem + hydroMem + distMem;
    cpuMem   = nbMem + mpiMem + hydroMem;
    if( proc == master ){
        if( cpuMem >= GB )
            fprintf( stdout, "Allocated %.4f GB or memory on host.\n", cpuMem/GB );
        if( cpuMem < GB )
            fprintf( stdout, "Allocated %.4f MB or memory on host.\n", cpuMem/MB );
        if( gpuMem >= GB )
            fprintf( stdout, "Allocated %.4f GB or memory on device.\n", gpuMem/GB );
        if( gpuMem < GB )
            fprintf( stdout, "Allocated %.4f MB or memory on device.\n", gpuMem/MB );
    }

//====================== INITIALIZE DATA =======================================
    init( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );
    logSave( devName, devMem, nb_east, nb_west, nb_north, nb_south, 
             phi, rho, ux, uy, uz );
    RELAX = 1;

    stats( phi, ux, uy, uz );

    vtkSave( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );

    cudaMemcpy( nb_east_d,  nb_east,  sizeof(int)*gridSize, cudaMemcpyHostToDevice ); 
    cudaMemcpy( nb_west_d,  nb_west,  sizeof(int)*gridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( nb_north_d, nb_north, sizeof(int)*gridSize, cudaMemcpyHostToDevice ); 
    cudaMemcpy( nb_south_d, nb_south, sizeof(int)*gridSize, cudaMemcpyHostToDevice );  

    cudaMemcpy( phi_d, phi, sizeof(float)*gridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( rho_d, rho, sizeof(float)*gridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( ux_d,  ux,  sizeof(float)*gridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( uy_d,  uy,  sizeof(float)*gridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( uz_d,  uz,  sizeof(float)*gridSize, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( dcol, &gridSize, sizeof(int), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( NX,   &NX_h,  sizeof(int), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( NY,   &NY_h,  sizeof(int), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( NZ,   &NZ_h,  sizeof(int), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( NXY,  &NXY_h, sizeof(int), 0, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( K0, &Wn0, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( K1, &Wn1, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( K2, &Wn2, sizeof(float), 0, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( KC0, &WnC0, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( KC1, &WnC1, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( KC2, &WnC2, sizeof(float), 0, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( alpha4_d,    &alpha4,    sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( kappa_d,     &kappa,     sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( Gamma_d,     &Gamma,     sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( eta_d,       &eta,       sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( eta2_d,      &eta2,      sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( phiStarSq_d, &phiStarSq, sizeof(float), 0, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( invEta2_d,      &invEta2,      sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( invTauPhi_d,    &invTauPhi,    sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( invTauPhiOne_d, &invTauPhiOne, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( invTauRhoOne_d, &invTauRhoOne, sizeof(float), 0, cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( zl_d, &zl, sizeof(int), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( zu_d, &zu, sizeof(int), 0, cudaMemcpyHostToDevice );

    init_f <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                    nb_south_d, phi_d, f_0_d, f_1_d, 
                                    f_2_d, f_3_d, f_4_d, f_5_d, f_6_d );

    init_g <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                    nb_south_d, phi_d, rho_d, g_0_d,
                                    g_1_d, g_2_d, g_3_d, g_4_d, g_5_d, g_6_d,
                                    g_7_d, g_8_d, g_9_d, g_10_d, g_11_d, g_12_d,
                                    g_13_d, g_14_d, g_15_d, g_16_d, g_17_d, 
                                    g_18_d );

    if( proc == master ) fprintf( stdout, "Data initialization completed.\n" );
    cudaThreadSynchronize();
    cudaEventRecord( stopSetup, 0 );
    cudaEventSynchronize( stopSetup );

//====================== INTERFACE RELAXATION LOOP =============================
    RELAX_GRAV = 0.f;
    cudaMemcpyToSymbol( grav_d, &RELAX_GRAV, sizeof(float), 0, cudaMemcpyHostToDevice );
    for( step = 1; step <= relaxStep; step++){

        update_phi <<<dimgrid,dimblock>>> ( phi_d, f_0_d, f_1_d, f_2_d, f_3_d,
                                            f_4_d, f_5_d, f_6_d );

        update_rho <<<dimgrid,dimblock>>> ( rho_d, g_0_d, g_1_d, g_2_d, g_3_d, 
                                            g_4_d, g_5_d, g_6_d, g_7_d, g_8_d, 
                                            g_9_d, g_10_d, g_11_d, g_12_d, 
                                            g_13_d, g_14_d, g_15_d, g_16_d, 
                                            g_17_d, g_18_d );

        update_velocity <<<dimgrid,dimblock>>> ( rho_d, ux_d, uy_d, uz_d, g_1_d, 
                                                 g_2_d, g_3_d, g_4_d, g_5_d, 
                                                 g_6_d, g_7_d, g_8_d, g_9_d, 
                                                 g_10_d, g_11_d, g_12_d, g_13_d, 
                                                 g_14_d, g_15_d, g_16_d, g_17_d,
                                                 g_18_d );

        // Update phi values in the ghost nodes for the Laplacian calculation 
        pack_mpi_phi <<<dimgrid,dimblock>>> ( topF_snd_d, botF_snd_d, phi_d );
   	
        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        mpiUpdate_phi( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_phi <<<dimgrid,dimblock>>> ( topF_rcv_d, botF_rcv_d, phi_d );

        collision_f <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                             nb_south_d, phi_d, rho_d, ux_d, 
                                             uy_d, uz_d, f_0_d, f_1_d, f_2_d,
                                             f_3_d, f_4_d, f_5_d, f_6_d );

        collision_g <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                             nb_south_d, phi_d, rho_d, ux_d, 
                                             uy_d, uz_d,  g_0_d, g_1_d, g_2_d, 
                                             g_3_d, g_4_d, g_5_d, g_6_d,
                                             g_7_d, g_8_d, g_9_d, g_10_d,
                                             g_11_d, g_12_d, g_13_d, 
                                             g_14_d, g_15_d, g_16_d, 
                                             g_17_d, g_18_d );

        // Before streaming f we need to make sure that the outward f components
        // in the ghost nodes are correct, because they are required in the 
        // smoothing step introduced by ZSC.
        // PERFORMANCE NOTE: see effect of skipping smoothing step for these 
        //                   nodes in order to save one MPI exchange.
        pack_mpi_f <<<dimgrid,dimblock>>> ( topF_snd_d, botF_snd_d, f_5_d, f_6_d );
   	
        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        mpiUpdate_f( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_f <<<dimgrid,dimblock>>> ( topF_rcv_d, botF_rcv_d, f_5_d, f_6_d );

        stream_f <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d,
                                          nb_south_d, f_0_d, f_1_d, f_2_d,
                                          f_3_d, f_4_d, f_5_d, f_6_d );

        stream_g <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d,
                                          nb_south_d, g_0_d,  g_1_d, g_2_d,
                                          g_3_d, g_4_d, g_5_d, g_6_d, g_7_d,
                                          g_8_d, g_9_d, g_10_d, g_11_d, g_12_d,
                                          g_13_d, g_14_d, g_15_d, g_16_d, 
                                          g_17_d, g_18_d );

        // Think carefully how to do this block partition to avoid doing
        // multiple if branches all over the place
        // This is the part that really hurts performance: copying to the host 
        // and back to do the MPI exchange. We should use streams and asynchronous
        // exchange of the boundaries only so that we can work on the core nodes
        // while the host/device and host/host communication is going on. 
        pack_mpi <<<dimgrid,dimblock>>> ( top_snd_d, bot_snd_d,
                                          f_5_d,  f_6_d,  g_5_d,  g_6_d,  
                                          g_11_d, g_12_d, g_13_d, g_14_d, 
                                          g_15_d, g_16_d, g_17_d, g_18_d );

        cudaThreadSynchronize();
        cudaMemcpy( top_snd, top_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );
        cudaMemcpy( bot_snd, bot_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );
        mpiUpdate( top_snd, bot_snd, top_rcv, bot_rcv );

        cudaMemcpy( top_rcv_d, top_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        cudaMemcpy( bot_rcv_d, bot_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        unpack_mpi <<<dimgrid,dimblock>>> ( top_rcv_d, bot_rcv_d,
                                            f_5_d,  f_6_d,  g_5_d,  g_6_d,  
                                            g_11_d, g_12_d, g_13_d, g_14_d, 
                                            g_15_d, g_16_d, g_17_d, g_18_d );

        if( (step%stat) == 0 ){
            cudaThreadSynchronize();
            cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            stats( phi, ux, uy, uz );
        }
        if( (step%save) == 0 ){
            cudaThreadSynchronize();
            cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( rho, rho_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            vtkSave( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );
        }
    }
    logSave( devName, devMem, nb_east, nb_west, nb_north, nb_south, 
             phi, rho, ux, uy, uz );
    if( proc == master ) fprintf( stdout, "Relaxation run completed.\n" );
    RELAX = 0;
    cudaMemcpyToSymbol( grav_d, &grav, sizeof(float), 0, cudaMemcpyHostToDevice );

//====================== SAVE RELAXED CONFIGURATION ============================
    cudaThreadSynchronize();
    cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( rho, rho_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    step = 0;
    stats( phi, ux, uy, uz );
    vtkSave( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );
    cudaEventRecord( stopRelax, 0 );
    cudaEventSynchronize( stopRelax );

//====================== MAIN CALCULATION LOOP =================================
    if( proc == master ) printf( "Starting evolution run ...\n" );
    for( step = 1; step < maxStep; step++){

        update_phi <<<dimgrid,dimblock>>> ( phi_d, f_0_d, f_1_d, f_2_d, f_3_d,
                                            f_4_d, f_5_d, f_6_d );

        update_rho <<<dimgrid,dimblock>>> ( rho_d, g_0_d, g_1_d, g_2_d, g_3_d, 
                                            g_4_d, g_5_d, g_6_d, g_7_d, g_8_d, 
                                            g_9_d, g_10_d, g_11_d, g_12_d, 
                                            g_13_d, g_14_d, g_15_d, g_16_d, 
                                            g_17_d, g_18_d );

        update_velocity <<<dimgrid,dimblock>>> ( rho_d, ux_d, uy_d, uz_d, g_1_d, 
                                                 g_2_d, g_3_d, g_4_d, g_5_d, 
                                                 g_6_d, g_7_d, g_8_d, g_9_d, 
                                                 g_10_d, g_11_d, g_12_d, g_13_d, 
                                                 g_14_d, g_15_d, g_16_d, g_17_d,
                                                 g_18_d );

        // Update phi values in the ghost nodes for the Laplacian calculation 
        pack_mpi_phi <<<dimgrid,dimblock>>> ( topF_snd_d, botF_snd_d, phi_d );

        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        mpiUpdate_phi( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_phi <<<dimgrid,dimblock>>> ( topF_rcv_d, botF_rcv_d, phi_d );

        collision_f <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d,
                                             nb_south_d, phi_d, rho_d, ux_d, 
                                             uy_d, uz_d, f_0_d, f_1_d, f_2_d,
                                             f_3_d, f_4_d, f_5_d, f_6_d );

        collision_g <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                             nb_south_d, phi_d, rho_d, ux_d, 
                                             uy_d, uz_d,  g_0_d, g_1_d, g_2_d, 
                                             g_3_d, g_4_d, g_5_d, g_6_d,
                                             g_7_d, g_8_d, g_9_d, g_10_d,
                                             g_11_d, g_12_d, g_13_d, 
                                             g_14_d, g_15_d, g_16_d, 
                                             g_17_d, g_18_d );

        // Before streaming f we need to make sure that the outward f components
        // in the ghost nodes are correct, because they are required in the 
        // smoothing step introduced by ZSC.
        // PERFORMANCE NOTE: see effect of skipping smoothing step for these 
        //                   nodes in order to save one MPI exchange.
        pack_mpi_f <<<dimgrid,dimblock>>> ( topF_snd_d, botF_snd_d, f_5_d, f_6_d );

        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        mpiUpdate_f( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_f <<<dimgrid,dimblock>>> ( topF_rcv_d, botF_rcv_d, f_5_d, f_6_d );


        stream_f <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d, 
                                          nb_south_d, f_0_d, f_1_d, f_2_d, 
                                          f_3_d, f_4_d, f_5_d, f_6_d );

        stream_g <<<dimgrid,dimblock>>> ( nb_east_d, nb_west_d, nb_north_d,
                                          nb_south_d, g_0_d, g_1_d, g_2_d,  
                                          g_3_d, g_4_d, g_5_d, g_6_d, g_7_d, 
                                          g_8_d, g_9_d, g_10_d, g_11_d, g_12_d,
                                          g_13_d, g_14_d, g_15_d, g_16_d, 
                                          g_17_d, g_18_d );

        // Think carefully how to do this block partition to avoid doing
        // multiple if branches all over the place
        // This is the part that really hurts performance: copying to the host 
        // and back to do the MPI exchange. We should use streams and asynchronous
        // exchange of the boundaries only so that we can work on the core nodes
        // while the host/device and host/host communication is going on. 
        pack_mpi <<<dimgrid,dimblock>>> ( top_snd_d, bot_snd_d,
                                          f_5_d,  f_6_d,  g_5_d,  g_6_d,  
                                          g_11_d, g_12_d, g_13_d, g_14_d, 
                                          g_15_d, g_16_d, g_17_d, g_18_d );

        cudaThreadSynchronize();
        cudaMemcpy( top_snd, top_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );
        cudaMemcpy( bot_snd, bot_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );
        mpiUpdate( top_snd, bot_snd, top_rcv, bot_rcv );

        cudaMemcpy( top_rcv_d, top_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        cudaMemcpy( bot_rcv_d, bot_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        unpack_mpi <<<dimgrid,dimblock>>> ( top_rcv_d, bot_rcv_d,
                                            f_5_d,  f_6_d,  g_5_d,  g_6_d,  
                                            g_11_d, g_12_d, g_13_d, g_14_d, 
                                            g_15_d, g_16_d, g_17_d, g_18_d );


        if( (step%stat) == 0 ){
            cudaThreadSynchronize();
            cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            stats( phi, ux, uy, uz );
        }
        if( (step%save) == 0 ){
            cudaThreadSynchronize();
            cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( rho, rho_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
            vtkSave( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );
        }

    }
    if( proc == master ) fprintf( stdout, "Evolution run completed.\n" );
    cudaThreadSynchronize();
    cudaEventRecord( stopMain, 0 );
    cudaEventSynchronize( stopMain );	
    cudaEventElapsedTime( &setupTime,   start,     stopSetup );
    cudaEventElapsedTime( &relaxTime,   stopSetup, stopRelax );
    cudaEventElapsedTime( &mainTime,    stopRelax, stopMain );
    cudaEventElapsedTime( &elapsedTime, start,     stopMain );
    if( proc == master ){
        printf( "Setup time      : %6.3f ms\n", setupTime );
        printf( "Relaxation time : %6.3f ms\n", relaxTime );
        printf( "Evolution time  : %6.3f ms\n", mainTime );
        printf( "Total time      : %6.3f ms\n", elapsedTime );
        printf( "Time for lb step: %6.3f ms\n", mainTime/( 1.f*maxStep ) );
    }

    // Save final configuration
    cudaMemcpy( phi, phi_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( rho, rho_d, sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( ux,  ux_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( uy,  uy_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy( uz,  uz_d,  sizeof(float)*gridSize, cudaMemcpyDeviceToHost );
    logSave( devName, devMem, nb_east, nb_west, nb_north, nb_south, 
             phi, rho, ux, uy, uz );
    stats( phi, ux, uy, uz );
    vtkSave( nb_east, nb_west, nb_north, nb_south, phi, rho, ux, uy, uz );

//====================== Free host dynamic arrays ==============================
    free( phi );       free( rho );
    free( nb_north );  free( nb_south );
    free( nb_east );   free( nb_west );   free( ux );  free( uy );  free( uz );

//====================== Free device dynamic arrays ==============================
    cudaFree( nb_north_d ); cudaFree( nb_south_d );
    cudaFree( nb_east_d );   cudaFree( nb_west_d );

    cudaFree( phi_d ); cudaFree( rho_d ); 
    cudaFree( ux_d );  cudaFree( uy_d );  cudaFree( uz_d );

    cudaFree( f_0_d ); cudaFree( f_1_d ); cudaFree( f_2_d ); cudaFree( f_3_d );
    cudaFree( f_4_d ); cudaFree( f_5_d ); cudaFree( f_6_d );

    cudaFree( g_0_d );  cudaFree( g_1_d );  cudaFree( g_2_d );
    cudaFree( g_3_d );  cudaFree( g_4_d );  cudaFree( g_5_d );
    cudaFree( g_6_d );  cudaFree( g_7_d );  cudaFree( g_8_d );
    cudaFree( g_9_d );  cudaFree( g_10_d ); cudaFree( g_11_d );
    cudaFree( g_12_d ); cudaFree( g_13_d ); cudaFree( g_14_d );
    cudaFree( g_15_d ); cudaFree( g_16_d ); cudaFree( g_17_d ); 
    cudaFree( g_18_d );

    MPI_Finalize();

    return 0;
}



