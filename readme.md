# Kompilacja cpp
```
g++ -O3 -pthread main.cpp -o main
```
# Kompilacja cuda
```
nvcc -O3 -std=c++14 main.cu -o main
```
# Profilowanie na CPU - kompilacja
```
g++ -O3 -g -pthread cpu_profiler.cpp -o cpu_profiler
```
# Profilowanie na CPU - odpalenie
```
valgrind --tool=callgrind ./cpu_profiler
```
# Odpalenie
```
./main nazwa_pliku
```