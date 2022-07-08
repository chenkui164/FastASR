SRC_PATH = ./src
DIRS = $(shell find $(SRC_PATH) -maxdepth 3 -type d)
SRCS_CXX += $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
VPATH = $(DIRS)
OBJECTS=$(addprefix obj/,$(notdir $(SRCS_CXX:.cpp=.o)))
# OBJECTS=$(SRCS_C:.c=.o)
DEPENDS=$(OBJECTS:.o=.d)
CXXFLAGS+=-O3 -std=c++11 
# CXXFLAGS+=-Wall -fdiagnostics-color=auto -std=c++11 -fext-numeric-literals -O2 -pthread -fopenmp -Wl,-rpath-link=/home/ck/work/am_msk/libs/lib
# CXXFLAGS+=-Wall -fdiagnostics-color=auto -std=c++11 -fext-numeric-literals -O2 -pthread -fopenmp -Wl,-rpath-link=/home/ck/work/am_msk/libs/lib

ARM_CXX=/home/ck/work/am_msk/cross-pi-gcc-8.3.0-2/bin/arm-linux-gnueabihf-g++
PC_CXX=g++

CXX=$(PC_CXX)

LDLIBS=-lm -lfftw3f -lopenblas

# LDPATH+=-L/home/ck/work/am_msk/libs/lib

# INCLUDEPATH+=-I /home/ck/work/am_msk/libs/include


# test:
# 	$(OBJECTS)


fastasr : $(OBJECTS)
	$(CXX) -o fastasr $(CXXFLAGS) $(OBJECTS) $(LDLIBS) $(LDPATH)
obj/%.o:%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(INCLUDEPATH) $(PREDEFINE)

obj/%.d:%.cpp
	@set -e; rm -f $@; $(CC) -MM $< $(INCLUDEFLAGS) > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

include $(DEPENDS)



.PHONY : clean install

install : fastasr
	expect ./download.ext

clean:
	@rm -f obj/*
	@rm -f fastasr
