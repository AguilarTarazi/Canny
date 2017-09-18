mpi:
	mpicc -fopenmp -o canny_mpi canny_mpi2.c -lm
	mpirun -np 4 canny_mpi imagen.pgm 1 0.3 0.8
	diff -s -q imagen-mpi.pgm im_serial.pgm

runp:
	gcc -fopenmp -o canny_paralelo canny_paralelo.c -pg -lm
	./canny_paralelo imagen.pgm 1 0.3 0.8
	diff -s -q im_paralelo.pgm im_serial.pgm
	# convert im_paralelo.pgm im_paralelo.jpg

runs:
	gcc -o canny  canny.c -pg -lm
	./canny imagen.pgm 1 0.3 0.8
	diff -s -q imagen-mpi.pgm im_serial.pgm
	# convert im_serial.pgm im_serial.jpg

clean:
	rm --o canny
	rm --o canny_paralelo
	rm --o canny_mpi
	rm im_paralelo.pgm
	rm im_serial.pgm
	rm imagen-mpi.pgm

run:
	gcc -o canny  canny.c -pg -lm
	gcc -fopenmp -o canny_paralelo canny_paralelo.c -pg -lm
	./canny imagen.pgm 5 0.3 0.8
	./canny_paralelo imagen.pgm 5 0.3 0.8
	diff -s -q im_paralelo.pgm im_serial.pgm
	# convert im_paralelo.pgm im_paralelo.jpg
	# convert im_serial.pgm im_serial.jpg
