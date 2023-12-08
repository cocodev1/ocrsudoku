all:
	gcc matrix.c csv.c img.c train.c cli.c -lSDL2 -lSDL2_image -o cli -lm
