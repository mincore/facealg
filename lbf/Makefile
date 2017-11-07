
LBFSRC=./src
LINSRC=./3rdparty/liblinear

CXXFLAGS+=-DNDEBUG
CXXFLAGS+=-fPIC -Wall -fopenmp
CXXFLAGS+=-I$(LBFSRC) -I$(LINSRC)/..

LDFLAGS+=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect

OBJS=.obj/ssd_landmark.o\
     .obj/lbf/lbf.o\
     .obj/lbf/common.o\
     .obj/lbf/rf.o\
     .obj/linear/daxpy.o\
     .obj/linear/ddot.o\
     .obj/linear/dnrm2.o\
     .obj/linear/dscal.o\
     .obj/linear/tron.o\
     .obj/linear/linear.o\

$(shell mkdir -p .obj/lbf .obj/linear)

liblbf.a: $(OBJS)
	@ar cr ../lib/$@ $^

.obj/lbf/%.o : $(LBFSRC)/%.cpp
	@g++ $< -c -o $@ $(CXXFLAGS)

.obj/linear/%.o : $(LINSRC)/%.c
	@g++ $< -c -o $@ $(CXXFLAGS)

.obj/linear/%.o : $(LINSRC)/%.cpp
	@g++ $< -c -o $@ $(CXXFLAGS)

.obj/%.o : %.cpp
	@g++ $< -c -o $@ $(CXXFLAGS)

clean:
	@rm -rf .obj ../lib/liblbf.a
