//-------------------------------------------------------------------------------
// Function : vgrid
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Create processor grid for MPI exchanges.
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

int vgrid()
{
    int complete, direction, partial, shift, mpi_coords;

    // Create the new virtual connectivity grid
    MPI_Cart_create( MPI_COMM_WORLD, MPI_DIM, &nprocs, &PERIODIC, REORDER, &MPI_COMM_VGRID );

    // Get this processor ID and coordinates within the virtual grid
    MPI_Comm_rank( MPI_COMM_VGRID, &vproc );
    MPI_Cart_coords( MPI_COMM_VGRID, vproc, MPI_DIM, &mpi_coords );

    //  Partitioning in the x direction
    complete = ( zmax - zmin ) / nprocs;
    partial  = ( zmax - zmin ) - complete*nprocs;
    if( mpi_coords + 1 <= partial ){
      zl = zmin + ( complete + 1 )*mpi_coords;
      zu = zmin + ( complete + 1 )*( mpi_coords + 1 ) - 1;
    }
    else{
      zl = zmin + complete*mpi_coords + partial;
      zu = zmin + complete*( mpi_coords + 1 ) + partial - 1;
    }
    if( ( ( mpi_coords + 1 ) % nprocs ) == 0 ) zu = zu + 1;

    // Modify limits to include ghost layers
    zlg = zl - 1;
    zug = zu + 1;

    //-------- Determine neighbours of this processor -------------------------------
    // MPI_CART counts dimensions using 0-based arithmetic so that
    // direction = 0 -> x  |  direction = 1 -> y  |  direction = 2 -> z
    shift     = 1;
    direction = 0;
    MPI_Cart_shift( MPI_COMM_VGRID, direction, shift, &bot, &top );

    return 0;
}

