.PHONY: all lib test clean

EXTERNAL_INCLUDES=include/externals

CXX=g++
CXXFLAGS=-Wall -Werror -pedantic -std=c++17 -Iinclude -I$(VULKAN_SDK)/include -I$(EXTERNAL_INCLUDES) -ggdb -Wno-unused-function -Wfatal-errors
CXXFLAGS_TEST=-Wall -Werror -pedantic -std=c++17 -Iinclude -ggdb
LDFLAGS=
LDFLAGS_TEST=-L. -lvkc -lglfw -L$(VULKAN_SDK)/lib -lvulkan -lm

# TODO: modify for cross-platform
OUTLIB=libvkc.so

HEADERS = include/vkc/vkc.hpp include/vkc/vkc_internal_types.hpp include/vkc/vkc_types.hpp include/vkc/libcommon.hpp
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
	$(CXX) $(CXXFLAGS_TEST) $< $(LDFLAGS_TEST) -o $@

test: test_1.out

clean:
	rm *.o libvkc.so
	rm -rf *.out
