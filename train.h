//
// Created by simon on 10/30/2023.
//

#include "matrix.h"

#ifndef NNXOR_TRAIN_H
#define NNXOR_TRAIN_H


matrix* Relu(matrix* m);
matrix* dRelu(matrix* m);
matrix* expM(matrix* m);
matrix* sum(matrix* m);

matrix* one_hot_y(matrix* m);

double get_accuracy(matrix* prediction, matrix* y);
matrix* get_prediction(matrix* a2);


matrix** descent(matrix* x, matrix* y, double alpha, int iteration);

#endif //NNXOR_TRAIN_H
