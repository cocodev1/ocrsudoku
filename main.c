#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csv.h"
#include "matrix.h"
#include "trainXor.h"
#include "train.h"


int main() {
    //char* path = "mnist_train.csv";
    //matrix** m = csvToMatrix(path);
    matrix** dataset = csvToMatrix("mnist_train.csv");
    matrix* imgx = *(dataset + 0);
    matrix* imgy = *(dataset + 1);
    imgx = transpose(imgx);
    imgy = transpose(imgy);
    imgx = sclalar(imgx, (double) 1 / 255);
    matrix* one_hot = one_hot_y(imgy);

    descent(imgx, one_hot, 0.1, 10000);

    return 0;
}
