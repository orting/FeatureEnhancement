CPPFLAGS = -Wall -Wextra -Werror -pedantic -std=c++11 -O2 -g

MY_INC_DIR = include
MY_LIB_DIR = lib

LIB_DIRS = -L${MY_LIB_DIR}
LIBS =  -lfftw3_threads\
	-lfftw3\
	-lm\
	-lpthread

INCLUDE = -I${MY_INC_DIR}

OBJECTS = lib/Volume.o\
	  lib/VolumeList.o\
	  lib/Transforms.o\
	  lib/Gauss.o\
	  lib/AutomaticFilter.o\
	  lib/Util.o\
	  lib/FeatureMeasure.o\
	  lib/Filter.o

test: test.o ${OBJECTS}
	@echo g++ $< ${OBJECTS} ${LIB_DIRS} ${LIBS} -o $@
	@g++ $< ${OBJECTS} ${LIB_DIRS} ${LIBS} -o $@

objects: ${OBJECTS}

%.o: %.cpp
	@echo g++ -c ${INCLUDE} ${CPPFLAGS} $< -o $@
	@g++ -c ${INCLUDE} ${CPPFLAGS} $< -o $@
