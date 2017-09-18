mpi:
	mpicc -fopenmp -o canny_mpi canny_mpi2.c -lm
	mpirun -np 4 canny_mpi imagen.pgm 1 0.3 0.8
	diff -s -q imagen_mpi.pgm imagen_serial.pgm

openmp:
	gcc -fopenmp -o canny_openmp canny_openmp.c -pg -lm
	./canny_openmp imagen.pgm 1 0.3 0.8
	diff -s -q imagen_openmp.pgm imagen_serial.pgm

serial:
	gcc -o canny  canny.c -pg -lm
	./canny imagen.pgm 1 0.3 0.8
	# diff -s -q imagen_openmp.pgm imagen_serial.pgm
	# diff -s -q imagen_mpi.pgm imagen_serial.pgm

clean:
	rm --o canny
	rm --o canny_openmp
	rm --o canny_mpi
	rm imagen_openmp.pgm
	rm imagen_serial.pgm
	rm imagen_mpi.pgm

run:
	gcc -o canny  canny.c -pg -lm
	gcc -fopenmp -o canny_openmp canny_openmp.c -pg -lm
	mpicc -fopenmp -o canny_mpi canny_mpi2.c -lm
	./canny imagen.pgm 5 0.3 0.8
	./canny_openmp imagen.pgm 5 0.3 0.8
	mpirun -np 4 canny_mpi imagen.pgm 1 0.3 0.8
	diff -s -q imagen_openmp.pgm imagen_serial.pgm
	diff -s -q imagen_mpi.pgm imagen_serial.pgm
	# convert im_paralelo.pgm im_paralelo.jpg
	# convert im_serial.pgm im_serial.jpg
