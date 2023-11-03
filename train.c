//
// Created by simon on 10/30/2023.
//
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <float.h>
#include "matrix.h"
#include "train.h"

double addScalar(double x, double y) {
    return x + y;
}

double substractScalar(double x, double y) {
    return x - y;
}

double divScalar(double x, double y) {
    return x / y;
}

double mulScalar(double x, double y) {
    return x * y;
}



double ReluScalar(double d) {
    if(d > 0) {
        return d;
    }
    return 0;
}

double dReluScalar(double d) {
    return d > 0;
}

matrix* Relu(matrix* m) {
    return fn(m, &ReluScalar);
}

matrix* dRelu(matrix* m) {
    return fn(m, &dReluScalar);
}


matrix* expM(matrix* m) {
    matrix* res = fn(m, exp);
    return res;
}

matrix* sum(matrix* m) {
    double* data = malloc(sizeof(double) * m->col);
    for(int i = 0; i < m->col; i++) {
        double curr = 0;
        for(int j = 0; j < m->row; j++) {
            curr += *(m->data + (m->col * j + i));
        }
        *(data + i) = curr;
    }
    return init(1, m->col, data);
}

double sumNoAxis(matrix* m) {
    double res = 0;
    for(int i = 0; i < m->row * m->col; i++) {
        res += *(m->data + i);
    }
    return res;
}

double addone (double x) {
    return x + 1;
}

matrix* softmax(matrix* m) {

    double* data = malloc(sizeof(double) * m->row * m->col);
    for(int i = 0; i < m->col*m->row; i++) {
        *(data + i) = *(m->data + i);
    }
    matrix* mm = init(m->row, m->col, data);
    matrix* mmm = sclalar(mm, (double) 1 / 1000000);
    matrix* expm = expM(mmm);
    matrix* summ = sum(expm);
    return fnmBroadcast(expm, summ, &divScalar);
}

matrix* one_hot_y(matrix* m) {
    if(m->row != 1) {
        errx(1, "row must be equal to 1");
    }
    matrix* res = zeros(10, m->col);
    for(int i = 0; i < m->col; i++) {
        int value = (int) *(m->data + i);
        *(res->data + (res->col * value + i)) = 1;
    }
    return res;
}

matrix* get_prediction(matrix* a2) {
    double* data = malloc(sizeof(double) * a2->col);
    for(int i = 0; i < a2->col; i++) {
        int max = 0;
        double maxval = 0;
        for(int j = 0; j < a2->row; j++) {
            if (*(a2->data + (a2->col * j + i)) > maxval) {
                maxval = *(a2->data + (a2->col * j + i));
                max = j;
            }
        }
        *(data + i) = max;
    }
    return init(1, a2->col, data);
}

double get_accuracy(matrix* prediction, matrix* y) {
    int sum = y->col;
    int goodres = 0;
    for(int i = 0; i < prediction->col; i++) {
        if(*(prediction->data+i) == *(y->data + i)) {
            goodres++;
        }
    }
    return (double) goodres / sum;
}

matrix** initParams() {
    matrix* w1 = initRandom(10, 784);
    matrix* b1 = initRandom(10, 1);
    matrix* w2 = initRandom(10, 10);
    matrix* b2 = initRandom(10, 1);
    matrix** res = malloc(sizeof(matrix*) * 4);
    *(res + 0) = w1;
    *(res + 1) = b1;
    *(res + 2) = w2;
    *(res + 3) = b2;
    return res;
}



matrix** foward(matrix* x, matrix** params) {
    matrix* w1 = *(params + 0);
    matrix* b1 = *(params + 1);
    matrix* w2 = *(params + 2);
    matrix* b2 = *(params + 3);

    matrix* intedot1 = dot(w1, x);
    matrix* z1 = fnmBroadcast(intedot1, b1, &addScalar);
    freem(intedot1);
    matrix* a1 = Relu(z1);

    matrix* inteddot2 = dot(w2, a1);
    matrix* z2 = fnmBroadcast(inteddot2, b2, &addScalar);
    freem(inteddot2);
    matrix* a2 = softmax(z2);
    matrix** res = malloc(sizeof(matrix*) * 4);
    *(res + 0) = z1;
    *(res + 1) = a1;
    *(res + 2) = z2;
    *(res + 3) = a2;
    return res;

}

matrix** backward(matrix* x, matrix* y, matrix** params, matrix** layer) {
    matrix* w1 = *(params + 0);
    matrix* b1 = *(params + 1);
    matrix* w2 = *(params + 2);
    matrix* b2 = *(params + 3);
    matrix* z1 = *(layer + 0);
    matrix* a1 = *(layer + 1);
    matrix* z2 = *(layer + 2);
    matrix* a2 = *(layer + 3);

    matrix* dz2 = fnm(a2, y, &substractScalar);
    matrix* a1T= transpose(a1);
    matrix* dw2 = sclalar(dot(dz2, a1T), (double)  1/ x->col);
    freem(a1T);

    double db2n = sumNoAxis(dz2) / x->col;
    matrix* db2 = zeros(b2->row, b2->col);
    for(int i = 0; i < b2->row * b2->col; i++) {
        *(db2->data + i) = db2n;
    }

    matrix* w2T = transpose(w2);
    matrix* intedot1 = dot(w2T, dz2);
    matrix* intedot2 = dRelu(z1);
    matrix* dz1 = fnm(intedot1, intedot2, &mulScalar);
    freem(w2T);
    freem(intedot1);
    freem(intedot2);

    matrix* xT = transpose(x);
    matrix* intedot3 = dot(dz1, xT);
    matrix* dw1 = sclalar(intedot3, (double) 1 / x->col);
    freem(xT);
    freem(intedot3);

    double db1n = sumNoAxis(dz1) / x->col;
    matrix* db1 = zeros(b1->row, b1->col);
    for(int i = 0; i < b1->row * b1->col; i++) {
        *(db1->data + i) = db1n;
    }
    matrix** res = malloc(sizeof(matrix*) * 4);
    *(res + 3) = db2;
    *(res + 2) = dw2;
    *(res + 1) = db1;
    *(res + 0) = dw1;

    return res;
}

matrix** update(matrix** params, matrix** gradient, double alpha) {
    matrix* intedot1 = sclalar(*(gradient + 0), alpha);
    matrix* nw1 = fnm(*(params + 0), intedot1, &substractScalar);
    freem(intedot1);

    matrix* intedot2 = sclalar(*(gradient + 1), alpha);
    matrix* nb1 = fnm(*(params + 1), intedot2, &substractScalar);
    freem(intedot2);

    matrix* intedot3 = sclalar(*(gradient + 2), alpha);
    matrix* nw2 = fnm(*(params + 2), intedot3, &substractScalar);
    freem(intedot3);

    matrix* intedot4 = sclalar(*(gradient + 3), alpha);
    matrix* nb2 = fnm(*(params + 3), intedot4, &substractScalar);
    freem(intedot4);

    matrix** res = malloc(sizeof(matrix*) * 4);
    *(res + 0) = nw1;
    *(res + 1) = nb1;
    *(res + 2) = nw2;
    *(res + 3) = nb2;
    return res;
}

matrix** descent(matrix* x, matrix* y, double alpha, int iteration) {
    matrix** params = initParams();
    for(int i = 0; i < iteration; i++) {
        matrix** layer = foward(x, params);
        if(i % 10 == 0) {
            matrix* a2 = *(layer + 3);
            matrix* pred = get_prediction(a2);
            double accuracy = get_accuracy(pred, y);
            printf("Step: %d Accuracy: %.2f%%\n", i, accuracy * 100);
        }
        matrix** gradient = backward(x, y, params, layer);
        matrix** nparams = update(params, gradient, alpha);

        freem(*(params + 0));
        freem(*(params + 1));
        freem(*(params + 2));
        freem(*(params + 3));
        freem(*(gradient + 0));
        freem(*(gradient + 1));
        freem(*(gradient + 2));
        freem(*(gradient + 3));
        freem(*(layer + 0));
        freem(*(layer + 1));
        freem(*(layer + 2));
        freem(*(layer + 3));
        free(params);
        free(gradient);
        free(layer);

        params = nparams;

    }
    return params;
}