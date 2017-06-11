all: fastfba

CCFLAGS+=--std=c99
CXXFLAGS+=-I src/compute/include/
CXXFLAGS+=`pkg-config --cflags eigen3`
CXXFLAGS+=-std=c++11
CXXFLAGS+=-DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION -DIMG_NO_FFTW -DIMG_NO_OMP
CXXFLAGS+=-O2

LDFLAGS+=-lOpenCL -lclFFT
# iio dependencies
LDFLAGS+=-lpng -ltiff -ljpeg
# ceres dependencies
LDFLAGS+=-lceres -lglog -llapack -lblas -lcholmod -lcxsparse -lpthread -fopenmp

OBJS=src/main.o src/image.o src/iio.o

fastfba: ${OBJS}
	${CXX} $^ -o $@ ${LDFLAGS}

clean:
	-rm ${OBJS}

