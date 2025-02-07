#ifndef TOOLS_H
#define TOOLS_H

void padding(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n);

void datacpy_f(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n);

void reload_f(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n);

void print_mat(float* arr, int m, int n);

void compare_mat(float* a, float* b, int m, int n);

void compare_vec(float* a, float* b, int size);

#endif