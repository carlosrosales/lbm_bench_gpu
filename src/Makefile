# Development
# MPICC   := nvcc -Xptxas -v -O0 -lm

# Production
MPICC := nvcc -Xptxas -v -O3 -use_fast_math -arch=sm_11 -lm

MPI_INC := /opt/openmpi/1.5/include
MPI_LIB := /opt/openmpi/1.5/lib

cudalb:
	$(MPICC) -I$(MPI_INC) -L$(MPI_LIB) -lmpi main.cu -o cudalb_mpi


