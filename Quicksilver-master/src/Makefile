SHELL = /bin/bash

###############################################################################
# Serial
###############################################################################
OPENMP_FLAGS = 
OPENMP_LDFLAGS =
OPTFLAGS = -g -O2

CXX=g++
CXXFLAGS = -std=c++11  $(OPTFLAGS) -Wpedantic
CPPFLAGS = $(OPENMP_FLAGS)
LDFLAGS = $(OPENMP_LDFLAGS) 


###############################################################################
# OpenMP
###############################################################################
# OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp
# OPENMP_LDFLAGS = -fopenmp
# OPTFLAGS = -g -O2

# CXX=g++
# CXXFLAGS = -std=c++11  $(OPTFLAGS) -Wpedantic
# CPPFLAGS = $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 

###############################################################################
### MPI and OpenMP 
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
# NVCC
###############################################################################
# OPENMP_FLAGS = 
# OPENMP_LDFLAGS = 
# OPTFLAGS =

# CXX=nvc++
# CXXFLAGS =  $(OPTFLAGS)
# CPPFLAGS = $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 

###############################################################################
# OpenAcc CPU
###############################################################################
# OPENMP_FLAGS = -DHAVE_OPENACC 
# OPENMP_LDFLAGS = 
# OPTFLAGS =

# CXX=nvc++
# CXXFLAGS = -acc=multicore $(OPTFLAGS)
# CPPFLAGS = $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 

###############################################################################
# OpenAcc GPU
###############################################################################
# OPENMP_FLAGS = -DHAVE_OPENACC_GPU
# OPENMP_LDFLAGS = 
# OPTFLAGS =

# CXX=nvc++
# CXXFLAGS = -acc -gpu=cc75 $(OPTFLAGS)
# CPPFLAGS = $(OPENMP_FLAGS)
# LDFLAGS = $(OPENMP_LDFLAGS) 

##############################################################################
# Cuda
##############################################################################
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


################################################################################
### Below here, it is pitch black.                                           ###
### You are likely to be eaten by a grue.                                    ###
################################################################################


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
	rm -f *.o git_hash.hh git_vers.hh .depend load.map *.core *.optrpt qs

cleanall:
	rm -f *.o git_hash.hh git_vers.hh .depend load.map *.core *.optrpt *.out qs

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
