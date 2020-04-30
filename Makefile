.PHONY: all lib test clean

CXX=g++
CXXFLAGS=-Wall -Werror -pedantic -std=c++11 -Iinclude
LDFLAGS=

# TODO: modify for cross-platform
OUTLIB=libvkc.so

HEADERS = include/vkc/vkc.hpp
SOURCES = src/vkc_common.cpp
OBJS = $(addsuffix .o, $(basename $(notdir $(SOURCES))))

TEST_SOURCES = src/tests/test_1.cpp
EXE_TESTS = $(addsuffix .out, $(basename $(notdir $(TEST_SOURCES))))

%.o: src/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

all: lib test

lib: $(OUTLIB)

$(OUTLIB): $(OBJS)
	$(CXX) -shared -o $(OUTLIB) $^ $(LDFLAGS)

%.out: src/tests/%.cpp $(HEADER)
	$(CXX) $(CXXFLAGS) $< -L. -lvkc -o $@

test: test_1.out

clean:
	rm *.o libvkc.so
	rm -rf *.out
