Carlos Rosales-Fernandez
carlos@tacc.utexas.edu

DESCRIPTION
===========
This is a code that simulates multiphase flow using the Lattice Boltzmann 
Method. It si designed to simulate mostly bubbly flows. It consists of an 
initialization step which is run on the GPU, followed by two big 
loops which carry out a relaxation on the fluid interfaces without external 
forces and then an evolution loop with external forces present. In both loops 
the same steps are taken, with the only difference that in the 
evolution loop I add a forcing term inside the "collision" functions.

INPUTS
======
I include an example in the directory "input". You will need to modify the
 "discrete.in" and "properties.in" files in order to change domain sizes, etc.
 
Currently the MPI parallelization is very simple, doing a 1D partitioning along 
the z direction. The CUDA part uses a 2D grid with number of threads per block
specified in the input file "properties.in" as BLOCK_SIZE_X and BLOCK_SIZE_Y.
Using 32x8 threads has worked well for me in the past.

The domain size is given by (xmax, ymax, zmax) in the same file. Bubble position
and radius is specified in the "discrete.in" file. Since that does not affect 
performance I would not touch it unless you want to use even smaller domains 
than the one I send you (which is already very small).

STRUCTURE
=========
I think the structure of the code should be clear once you have a quick look at 
"main.cu". The main thing to consider is what happens in the calculation loops. 
I include a high level description below. As you can see I do a standard Copy to
 host followed by the MPI exchange and then a new copy to device. This is 
obviously the main bottleneck in the code.

I have not had enough time to do extensive optimization of the communication
part, but any advise/comments you have would be greatly appreciated.

This is what the structure of the calculation loops looks like:

Step 1: Do some calculations (all done int he GPU)

        update_phi <<<dimgrid,dimblock>>> 
        update_rho <<<dimgrid,dimblock>>> 
        update_velocity <<<dimgrid,dimblock>>> 

Step 2: Pack and exchange the values of Phi that I need in ghost nodes (I do not
        think there is any way to merge this exchange with any of the others)

        pack_mpi_phi <<<dimgrid,dimblock>>>
        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );

        mpiUpdate_phi( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_phi <<<dimgrid,dimblock>>> 

Step 3: Run the main LBM calculation, the collision of momentum and phase 
        distributions (all done in the GPU)

        collision_f <<<dimgrid,dimblock>>> 
        collision_g <<<dimgrid,dimblock>>> 

Step 4: Pack and exchange the f values which are needed for the relaxation step
        (this is needed before streaming to make sure we have stability at large
         density ratio cases)

        pack_mpi_f <<<dimgrid,dimblock>>>
        cudaThreadSynchronize();
        cudaMemcpy( topF_snd, topF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );
        cudaMemcpy( botF_snd, botF_snd_d, sizeof(float)*bufSize, cudaMemcpyDeviceToHost );

        mpiUpdate_f( topF_snd, botF_snd, topF_rcv, botF_rcv );

        cudaMemcpy( topF_rcv_d, topF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        cudaMemcpy( botF_rcv_d, botF_rcv, sizeof(float)*bufSize, cudaMemcpyHostToDevice );
        unpack_mpi_f <<<dimgrid,dimblock>>> ( topF_rcv_d, botF_rcv_d, f_5_d, f_6_d );

Step 5: Run the LBM streaming step (mostly memory moves)

        stream_f <<<dimgrid,dimblock>>>
        stream_g <<<dimgrid,dimblock>>>

        pack_mpi <<<dimgrid,dimblock>>> 
        cudaThreadSynchronize();
        cudaMemcpy( top_snd, top_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );
        cudaMemcpy( bot_snd, bot_snd_d, sizeof(float)*bufSize*6, cudaMemcpyDeviceToHost );

        mpiUpdate( top_snd, bot_snd, top_rcv, bot_rcv );

        cudaMemcpy( top_rcv_d, top_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        cudaMemcpy( bot_rcv_d, bot_rcv, sizeof(float)*bufSize*6, cudaMemcpyHostToDevice );
        unpack_mpi <<<dimgrid,dimblock>>>

Step 6: Save data to file if we have reached the required point int he simulation

Step 7: Go back to step 1 until the loop si completed.
