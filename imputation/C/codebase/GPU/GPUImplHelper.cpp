/*
 *
 * @author Marc Suchard
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>

void checkHostMemory(void* ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Unable to allocate some memory!\n");
        exit(-1);
    }
}

void printfVectorD(double* ptr,
                   int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

void printfVectorF(float* ptr,
                   int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

template <typename REAL>
void printfVector(REAL* ptr,
                  int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

void printfInt(int* ptr,
               int length) {
    fprintf(stderr, "[ %d", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %d", ptr[i]);
    fprintf(stderr, " ]\n");
}
