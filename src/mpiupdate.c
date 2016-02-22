//-------------------------------------------------------------------------------
// Function : mpiUpdate
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales-Fernandez [carlos.rosales.fernandez(at)gmail.com]
//-------------------------------------------------------------------------------
// Exchange outward pointing components of f and g after streaming step.
// This function requires 4 arguments.
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

int mpiUpdate( float *top_snd, float *bot_snd, 
               float *top_rcv, float *bot_rcv )
{
    int         TAG1 = 1, TAG2 = 2;
    MPI_Status  mpistat;
    MPI_Request MPI_REQ1, MPI_REQ2;

    MPI_Irecv(top_rcv, 6*bufSize, MPI_FLOAT, top, TAG1, MPI_COMM_VGRID, &MPI_REQ1 );
    MPI_Irecv(bot_rcv, 6*bufSize, MPI_FLOAT, bot, TAG2, MPI_COMM_VGRID, &MPI_REQ2 );

    MPI_Send( bot_snd, 6*bufSize, MPI_FLOAT, bot, TAG1, MPI_COMM_VGRID );
    MPI_Send( top_snd, 6*bufSize, MPI_FLOAT, top, TAG2, MPI_COMM_VGRID );

    MPI_Wait( &MPI_REQ1, &mpistat );
    MPI_Wait( &MPI_REQ2, &mpistat );
    
    return 0;
}
