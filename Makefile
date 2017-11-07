#/* ===================================================
# * Copyright (C) 2017 chenshuangping All Right Reserved.
# *      Author: (chenshuangping)mincore@163.com
# *    Filename: Makefile
# *     Created: 2017-05-31 13:48
# * Description: 
# * ===================================================
# */
CXX=@echo "cc $@";g++
NVCC=@echo "nvcc $@";/usr/local/cuda-8.0/bin/nvcc

CAFFE_ROOT=/home/wenge/caffe_ssd

BINDIR=bin
TARGET=$(BINDIR)/libalga.so
TEST=$(BINDIR)/test

CXXFLAGS=-g -Wall -fPIC -std=c++11
CXXFLAGS+=-Isrc -Iinclude
CXXFLAGS+=-I$(CAFFE_ROOT)/include
CXXFLAGS+=-I$(CAFFE_ROOT)/build/src
CXXFLAGS+=-I/usr/local/cuda/include
CXXFLAGS+=-DNDEBUG
CXXFLAGS+=-DKKEY=\"$(KKEY)\"
CXXFLAGS+=-DKIV=\"$(KIV)\"
CXXFLAGS+=-DKSLAT=\"$(KSLAT)\"

OBJS=$(patsubst %.cpp,%.o,$(shell find src -name *.cpp))
OBJS+=src/resize.o
TEST_OBJS=$(patsubst %.cpp,%.o,$(shell find test -name *.cpp))

LDFLAGS+=-L/usr/local/cuda/lib64 -lcublas -lcudart -fopenmp -lopenblas `pkg-config --libs opencv`
LDFLAGS+=-Wl,-rpath=.
LDFLAGS+=-Lbin -lcaffe

all: $(TARGET) test

$(TARGET): $(OBJS)
	@make -C lbf
	$(CXX) $^ -shared -fPIC -o $@ $(LDFLAGS) -lglog -lboost_system lib/liblbf.a lib/libpack.a

$(TEST): $(TEST_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS) -pthread -Lbin -lalga -lprotobuf

%.o:%.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS) -std=c++11

src/resize.o:src/resize.cu
	$(NVCC) -o $@ -c $< --compiler-options -fPIC -std=c++11 -Wno-deprecated-gpu-targets

.PHONY : test clean

test: $(TARGET) $(TEST)

clean:
	@make clean -C lbf
	@rm -f $(OBJS) $(TARGET) $(TEST) $(TEST_OBJS)
