main:main.c trainXor.c matrix.c train.c
	gcc -Wall -Wextra matrix.c trainXor.c train.c main.c -o main -lm