set -o errexit

rm -f test_matrixMultiply.o matrixMultiply.o 
g++ -o2 test_matrixMultiply.cpp matrixMultiply.cpp -o test_matrixMultiply.o -mavx -fopenmp
./test_matrixMultiply.o
# valgrind --tool=cachegrind ./test_matrixMultiply.o
