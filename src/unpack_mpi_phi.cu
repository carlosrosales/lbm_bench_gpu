//-------------------------------------------------------------------------------
// Function : unpack_mpi_phi
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Gathers MPI boundary data into two arrays only to reduce communication time 
// betweeen host and device. 
//
// Notice in this update we use the updates values of phi, which are needed in 
// the ghosts for the gradient and laplacian calculations in the collision step.
//
// This function has 3 arguments.
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

__global__ void unpack_mpi_phi( float *topF_rcv_d, float *botF_rcv_d,
                                float *phi_d )
{
    int   i, j;

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    phi_d[ gridId( i, j, 0    ) ] = botF_rcv_d[ i + NX*j ];
    phi_d[ gridId( i, j, NZ-1 ) ] = topF_rcv_d[ i + NX*j ];  

}
