BOOST_DIR = /usr/include
BOOST_LIB_DIR = /usr/lib
ITK_DIR = /home/silas/Dokumenter/Uddannelse/Datalogi/ITK320/include/InsightToolkit
ITK_LIB_DIR = /home/silas/Dokumenter/Uddannelse/Datalogi/ITK320/lib/InsightToolkit

MY_LIB_DIR = lib
MY_INC_DIR = include

# The rest should be fine...
CIMG_DIR = ${MY_INC_DIR}/cimg #/usr/include
PECHIN_DIR = ${MY_INC_DIR}/pechin/codes
ITK_COMMON_DIR = ${ITK_DIR}/Common
ITK_ALGORITHMS_DIR = ${ITK_DIR}/Algorithms
ITK_UTILITIES_DIR = ${ITK_DIR}/Utilities
ITK_GDCM_DIR = ${ITK_DIR}/gdcm/src
ITK_GDCM_DICT_DIR = ${ITK_DIR}/gdcm/Dicts
ITK_NUMERICS_DIR = ${ITK_DIR}/Numerics
ITK_NUMERICS_STATISTICS_DIR = ${ITK_NUMERICS_DIR}/Statistics
ITK_BASIC_FILTERS_DIR = ${ITK_DIR}/BasicFilters
ITK_IO_DIR = ${ITK_DIR}/IO
ITK_SPATIAL_OBJECT_DIR = ${ITK_DIR}/SpatialObject
ITK_EXPAT_DIR = ${ITK_UTILITIES_DIR}/expat
ITK_ITKPNG_DIR = ${ITK_UTILITIES_DIR}/itkpng
ITK_ITKSYS_DIR = ${ITK_UTILITIES_DIR}/itksys
ITK_ITKTIFF_DIR = ${ITK_UTILITIES_DIR}/itktiff
ITK_ITKZLIB_DIR = ${ITK_UTILITIES_DIR}/itkzlib
ITK_METAIO_DIR = ${ITK_UTILITIES_DIR}/MetaIO
ITK_NRRDIO_DIR = ${ITK_UTILITIES_DIR}/NrrdIO
ITK_VXL_DIR = ${ITK_UTILITIES_DIR}/vxl
ITK_VXL_CORE_DIR = ${ITK_VXL_DIR}/core
ITK_VCL_DIR = ${ITK_VXL_DIR}/vcl

INCLUDE = -I${MY_INC_DIR}\
	  -I${CIMG_DIR}\
		  -I${PECHIN_DIR}\
          -I${BOOST_DIR}/include\
		  -I${ITK_DIR}\
		  -I${ITK_COMMON_DIR}\
		  -I${ITK_ALGORITHMS_DIR}\
		  -I${ITK_UTILITIES_DIR}\
		  -I${ITK_GDCM_DIR}\
		  -I${ITK_NUMERICS_DIR}\
		  -I${ITK_NUMERICS_STATISTICS_DIR}\
		  -I${ITK_BASIC_FILTERS_DIR}\
		  -I${ITK_IO_DIR}\
		  -I${ITK_SPATIAL_OBJECT_DIR}\
		  -I${ITK_EXPAT_DIR}\
		  -I${ITK_ITKPNG_DIR}\
		  -I${ITK_ITKTIFF_DIR}\
		  -I${ITK_ITKZLIB_DIR}\
		  -I${ITK_METAIO_DIR}\
		  -I${ITK_NRRDIO_DIR}\
		  -I${ITK_VXL_DIR}\
		  -I${ITK_VXL_CORE_DIR}\
		  -I${ITK_VCL_DIR}

LIB_DIRS = -L${ITK_LIB_DIR}\
           -L${BOOST_LIB_DIR}\
	   -L${MY_LIB_DIR}\

LIBS = -lX11\
       -lboost_filesystem\
       -lboost_program_options\
       -lboost_system\
       -lboost_thread\
	   -lITKCommon\
	   -lITKIO\
	   -lITKNumerics\
	   -litkvnl_algo\
	   -lITKMetaIO\
	   -litkvnl\
	   -litkzlib\
	   -litksys\
	   -litkgdcm\
	   -litkjpeg8\
	   -litkjpeg12\
	   -litkjpeg16\
	   -litkv3p_netlib\
	   -litktiff\
	   -lITKNrrdIO\
	   -litkpng\
	   -lITKniftiio\
	   -lITKDICOMParser\
	   -lITKznz\
	   -litkopenjpeg\
	   -lITKEXPAT\
	   -litkvcl\
	   -litkvnl_inst\
	   -lITKFEM\
	   -lITKSpatialObject\
	   -lITKAlgorithms\
	   -lITKBasicFilters\
	   -lITKStatistics\
	   -litkzlib\
	   -lITKCommon\
	   -luuid\
	   -lfftw3\
	   -lhdf5

vpath %.cpp ${PECHIN_DIR}

CPPFLAGS = -Wno-deprecated -frounding-math -fpermissive -Wall -Wextra -std=c++11 -O2 -g 

#TARGET=filter

OBJECTS=string_functions.o\
	random_generator.o\
	nr_eigen.o\
	file_functions.o\
	ced.o\
	lib/Util.o\
	lib/Feature.o\
	lib/Gauss.o\
	lib/AutomaticFilter.o\
	lib/SupervisedFilter.o\
	lib/Classifier.o


supervised: supervised.o ${OBJECTS}
	@echo Linking $@ from $<.
	@g++ $< ${OBJECTS} ${LIB_DIRS} ${LIBS} -o $@


automatic: automatic.o ${OBJECTS}
	@echo Linking $@ from $<.
	@g++ $< ${OBJECTS} ${LIB_DIRS} ${LIBS} -o $@

classify_training: classify_training.o string_functions.o random_generator.o nr_eigen.o file_functions.o ced.o
	@echo Linking $@ from $<.
	@g++ classify_training.o string_functions.o random_generator.o nr_eigen.o file_functions.o ced.o ${LIB_DIRS} ${LIBS} -o $@

test: test.o ${OBJECTS}
	@echo Linking $@ from $<.
	@g++ $< ${OBJECTS} ${LIB_DIRS} ${LIBS} -o $@

%.o: %.cpp
	@echo Compiling $@ from $<.
	@g++ -c ${INCLUDE} ${CPPFLAGS} $< -o $@

clean:
ifeq ($(OS), Windows_NT)
ifeq (,$(findstring .a,${TARGET}))
	@del ${TARGET} ${OBJECTS} $(OBJECTS:.o=.d)
else
	@del ${TARGET}.exe ${OBJECTS} $(OBJECTS:.o=.d)
endif
else
	@rm -f ${TARGET} ${OBJECTS} $(OBJECTS:.o=.d)
endif

.PHONY: clean

