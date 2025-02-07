#include "tools.h"
#include <iostream>

void compare(float* a, float* b, int m, int n){
  // compare
  int is_same = 1;
  for(int i = 0; i < m*n; i++){
    int x = i / n;
    int y = i % n;
    if(a[i] != b[i]){
      is_same = 0;
      std::cout << "(" << x << "," << y << ")  " << a[i] << "  " << b[i] << std::endl;
    }
  }
  if(is_same == 0){
    std::cout << "res:error!" << std::endl;
  }else{
    std::cout << "res:pass!" << std::endl;
  }
}

int main(){
    const int m = 1023;
    const int n = 1023;
    const int s_n = 1024;
    const int s_m = 1024;
    float * a = new float[m * n];
    float * b1 = new float[s_m * s_n];
    float * b2 = new float[s_m * s_n];

    for(int i = 0; i < m * n; i++){
        a[i] = i;
    }

    padding(b1, a, s_m, s_n, m, n);
    datacpy_f(b2, a, s_m, s_n, m, n);

    compare(b1, b2, s_m, s_n);
  
    delete[] a;
    delete[] b1;
    delete[] b2;


    return 0;
}