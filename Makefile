CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -fopenmp
LDFLAGS = -fopenmp

TARGET = gemm_bench
SRC = cpp/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
