SHELL = /bin/bash

###############################################################################
# Serial
###############################################################################
# OPENMP_FLAGS = 
# OPENMP_LDFLAGS =
# OPTFLAGS = -g -O2

# CXX=g++
# CXXFLAGS = -std=c++11  $(OPTFLAGS) -Wpedantic
# CPPFLAGS = $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 


###############################################################################
# Very simple GCC build with OpenMP but without MPI.  
# This works on a Macbook (if gcc is installed)
###############################################################################
OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp
OPENMP_LDFLAGS = -fopenmp
OPTFLAGS = -g -O2

CXX=g++
CXXFLAGS = -std=c++11  $(OPTFLAGS) -Wpedantic
CPPFLAGS = $(OPENMP_FLAGS)
LDFLAGS = $(OPENMP_LDFLAGS) 


###############################################################################
### GCC -- with MPI and OpenMP 
###############################################################################
# OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp
# OPENMP_LDFLAGS = -fopenmp
# MPI_FLAGS = -DHAVE_MPI
# OPTFLAGS = -g -O2

# CXX=mpicxx
# CXXFLAGS = -std=c++11 $(OPTFLAGS) -Wpedantic
# CPPFLAGS = $(MPI_FLAGS) $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 

###############################################################################
# OpenMP 4.5 on LLNL CORAL EA nodes
###############################################################################
# CUDA_PATH = /gpfs/warwick/scrtp/avon/eb/software/CUDAcore/11.1.1

# # Choose one of these compilers
# # CXX=mpiclang++-gpu
# CXX=mpixlC-gpu

# # Uncomment next line to run on CPU only (disable GPUs)
# OPENMP_OFFLOAD_FLAGS = -DUSE_OPENMP_NO_GPU 

# OPENMP_OFFLOAD_FLAGS += -DHAVE_OPENMP_TARGET -mcpu=power8  
# OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp ${OPENMP_OFFLOAD_FLAGS}
# OPTFLAGS = -O2

# CUDA_FLAGS = -I${CUDA_PATH}/include/
# CUDA_LDFLAGS = -L${CUDA_PATH}/lib64/ -lcuda -lcudart

# CXXFLAGS = -std=c++11 $(OPTFLAGS)
# CPPFLAGS = -DHAVE_MPI $(OPENMP_FLAGS) $(CUDA_FLAGS) 
# LDFLAGS  = $(CUDA_LDFLAGS)


###############################################################################
# Cuda
###############################################################################
# CUDA_PATH = /gpfs/warwick/scrtp/avon/eb/software/CUDAcore/11.1.1
# HOST_COMPILER = /gpfs/warwick/scrtp/avon/eb/software/OpenMPI/4.0.5-gcccuda-2020b/bin/mpicxx
# OPTFLAGS = -O2 
# CUDA_FLAGS = -I${CUDA_PATH}/include/
# CUDA_LDFLAGS = -L${CUDA_PATH}/lib64/ -lcuda -lcudart
# CXX=$(CUDA_PATH)/bin/nvcc
# CXXFLAGS = -DHAVE_CUDA -std=c++11 $(OPTFLAGS) -Xptxas -v 
# CXXFLAGS += -gencode=arch=compute_60,code=\"sm_60,compute_60\" 
# CXXFLAGS += --compiler-bindir=$(HOST_COMPILER)
# CPPFLAGS = -x cu -dc -DHAVE_MPI -DHAVE_ASYNC_MPI
# LDFLAGS  = $(CUDA_LDFLAGS) 
# #LDFLAGS += ${CUDA_PATH}/lib64/libnvToolsExt.so


################################################################################
### Below here, it is pitch black.                                           ###
### You are likely to be eaten by a grue.                                    ###
################################################################################

#GITVERS := -D'GIT_VERS="$(shell git log -n 1 |  grep Date   | awk -F " " '{print $$6 "-" $$3 "-" $$4 "-" $$5}')"'
#GITHASH := -D'GIT_HASH="$(shell git log -n 1 |  grep commit | awk -F " " '{print $$2}')"'
# GITVERS := "$(shell git log -n 1 |  grep Date   | awk -F " " '{print $$6 "-" $$3 "-" $$4 "-" $$5}')"
# GITHASH := "$(shell git log -n 1 |  grep commit | awk -F " " '{print $$2}')"

Quicksilver_EXE=qs

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .cc .o

.PHONY: DEFAULT clean distclean depend

# For development purposes, what is working now.
SOURCES= \
    CollisionEvent.cc \
    CoralBenchmark.cc \
    CycleTracking.cc \
    DecompositionObject.cc \
    DirectionCosine.cc \
	EnergySpectrum.cc \
    GlobalFccGrid.cc \
    GridAssignmentObject.cc \
    InputBlock.cc \
    MCT.cc \
    MC_Adjacent_Facet.cc \
    MC_Base_Particle.cc \
    MC_Domain.cc \
    MC_Facet_Crossing_Event.cc \
    MC_Fast_Timer.cc \
    MC_Load_Particle.cc \
    MC_Location.cc \
    MC_Particle_Buffer.cc \
    MC_RNG_State.cc \
    MC_Segment_Outcome.cc \
    MC_SourceNow.cc \
    MacroscopicCrossSection.cc \
    MeshPartition.cc \
    MonteCarlo.cc \
    MpiCommObject.cc \
    NuclearData.cc \
    Parameters.cc \
    ParticleVault.cc \
    ParticleVaultContainer.cc \
    PopulationControl.cc \
    SendQueue.cc \
    SharedMemoryCommObject.cc \
    Tallies.cc \
    cmdLineParser.cc \
    cudaFunctions.cc \
    initMC.cc \
    main.cc \
    parseUtils.cc \
    utils.cc \
    utilsMpi.cc 

CC_OBJECTS=$(SOURCES:.cc=.o)

DEFAULT: ${Quicksilver_EXE}

git_hash.hh:
	echo "#define GIT_HASH \"$(GITHASH)\" "> git_hash.hh

git_vers.hh:
	echo "#define GIT_VERS \"$(GITVERS)\" "> git_vers.hh

%.o: %.cc
	${CXX} ${CPPFLAGS} ${CXXFLAGS} -c $< -o $@

${Quicksilver_EXE}: git_hash.hh git_vers.hh ${CC_OBJECTS}  
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o ${Quicksilver_EXE} ${CC_OBJECTS}

clean:
	rm -f *.o git_hash.hh git_vers.hh .depend load.map *.core *.optrpt

cleanall:
	rm -f *.o git_hash.hh git_vers.hh .depend load.map *.core *.optrpt *.out

distclean: clean
	rm -f ${Quicksilver_EXE}   .depend.bak
	rm -rf html latex vtune*

.depend: $(SOURCES)
	@touch .depend
	@$(MAKE) --no-print-directory depend

depend:
	@echo "Rebuilding dependencies..."
	@makedepend -f .depend -Y. --$(CXXFLAGS) $(CPPFLAGS)-- $(SOURCES) 2> /dev/null

-include .depend
