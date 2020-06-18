build/mhg.o: c_code/mhg.c
	gcc -c -fPIC c_code/mhg.c -o build/mhg.o

build/logmhg.o: c_code/logmhg.c
	gcc -c -fPIC c_code/logmhg.c -o build/logmhg.o


build/library.so: build/mhg.o build/logmhg.o
	gcc build/mhg.o build/logmhg.o -shared -o build/library.so
