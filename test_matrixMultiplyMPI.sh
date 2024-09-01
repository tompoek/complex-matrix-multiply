set -o errexit

rm -f test_matrixMultiplyMPI.o matrixMultiplyMPI.o 
mpic++ -O2 test_matrixMultiplyMPI.cpp matrixMultiplyMPI.cpp -o test_matrixMultiplyMPI.o -mavx -fopenmp
mpirun -n 2 ./test_matrixMultiplyMPI.o
# valgrind --tool=cachegrind mpirun -n 2 ./test_matrixMultiplyMPI.o
