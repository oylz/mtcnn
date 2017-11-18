CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
AR = $(PREFIX)ar

CFLAGS += -Wall -O2
CFLAGS += -I$(TOPDIR)/include

#CXXFLAGS += -Wall -ggdb -std=c++11
CXXFLAGS += -Wall -O3 -std=c++11
CXXFLAGS += -I$(TOPDIR)/include

ARFLAGS = -rcv

CAFFE_ON = 0
MXNET_ON = 0
TF_ON = 1

#MTCNN_LDFLAGS = -L$(MTCNNLIBDIR) -Wl,--whole-archive -lmtcnn -Wl,--no-whole-archive
LDFLAGS += -L$(MTCNNLIBDIR) -lmtcnn

# opencv headers and libraries
CXXFLAGS += $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs opencv)

# mxnet settings
ifeq ($(MXNET_ON), 1)
   MXNET_INCS += -I$(MXNET_ROOT)/include
   MXNET_INCS += -I$(MXNET_ROOT)/dmlc-core/include
   MXNET_INCS += -I$(MXNET_ROOT)/nnvm/include

   LDFLAGS += -L$(MXNET_ROOT)/lib -lmxnet
   CXXFLAGS += $(MXNET_INCS) -Wno-sign-compare
endif

# caffe settings
ifeq ($(CAFFE_ON), 1)
   CAFFE_INCS := -I$(CAFFE_ROOT)/include
   CAFFE_INCS += -I$(CAFFE_ROOT)/distribute/include

   LDFLAGS += -L$(CAFFE_ROOT)/build/lib -lcaffe
   LDFLAGS += -lprotobuf -lboost_system -lglog
   CXXFLAGS += $(CAFFE_INCS) -DCPU_ONLY=1
endif

#  tensorflow settings
ifeq ($(TF_ON), 1)
   #TENSORFLOW_INCS += -I$(TENSORFLOW_ROOT)/include
   #LIBS+=-Wl,-rpath,$(TENSORFLOW_ROOT)/lib -L$(TENSORFLOW_ROOT)/lib -ltensorflow

   TENSORFLOW_INCS += -I/home/xyz/code1/tensorflow-1.4.0
   LIBS+=-Wl,-rpath,/home/xyz/code1/tensorflow-1.4.0/bazel-bin/tensorflow -L/home/xyz/code1/tensorflow-1.4.0/bazel-bin/tensorflow -ltensorflow

   CXXFLAGS +=$(TENSORFLOW_INCS) $(LIBS)
endif

# arm compute library setting
3LDFLAGS += -L$(ACL_ROOT)/build -larm_compute
LDFLAGS += -L$(ACL_ROOT)/build

%.os : %.cpp
	$(CXX) -fpic -shared $(CXXFLAGS) -c $< -o $@

%.i : %.c
	$(CC) $(CFLAGS) -E $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.d : %.c
	@$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed -e 's/\($(@F)\.o\)[ :]*/$(<D)\/\1 $(@D)\/$(@F) : /g' $@.$$$$ > $@ ; \
	$(RM) $@.$$$$;

%.i : %.cpp
	$(CXX) $(CXXFLAGS) -E $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.d : %.cpp
	@$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed -e 's/\($(@F)\.o\)[ :]*/$(<D)\/\1 $(@D)\/$(@F) : /g' $@.$$$$ > $@ ; \
	$(RM) $@.$$$$;
#.depens := $(patsubst %.o, %.d, $(lib_objs))
sinclude $(.depens)
