CXX ?= g++

CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: detect

detect: detect.o; $(CXX) $< -o $@ $(LDFLAGS) -g

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS) -g

clean: ; rm -f detect.o detect
