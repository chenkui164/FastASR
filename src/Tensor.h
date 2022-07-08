

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <malloc.h>
#include <stdint.h>
#include <string.h>

using namespace std;

template <typename T> class Tensor {
  private:
    void alloc_buff();
    void free_buff();

  public:
    T *buff;
    int size[4];
    int buff_size;
    Tensor(Tensor<T> *in);
    Tensor(int a);
    Tensor(int a, int b);
    Tensor(int a, int b, int c);
    Tensor(int a, int b, int c, int d);
    ~Tensor();
    void zeros();
    void shape();
    void disp();
    void add(float coe, Tensor<T> *in);
    void reload(Tensor<T> *in);
};

template <typename T> Tensor<T>::Tensor(int a) : size{1, 1, 1, a}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(int a, int b) : size{1, 1, a, b}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(int a, int b, int c) : size{1, a, b, c}
{

    alloc_buff();
}

template <typename T>
Tensor<T>::Tensor(int a, int b, int c, int d) : size{a, b, c, d}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(Tensor<T> *in)
{
    memcpy(size, in->size, 4 * sizeof(int));
    alloc_buff();
    memcpy(buff, in->buff, in->buff_size * sizeof(T));
}

template <typename T> Tensor<T>::~Tensor()
{
    free_buff();
}

template <typename T> void Tensor<T>::alloc_buff()
{
    buff_size = size[0] * size[1] * size[2] * size[3];
    buff = (T *)memalign(32, buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::free_buff()
{
    free(buff);
}

template <typename T> void Tensor<T>::zeros()
{
    memset(buff, 0, buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::shape()
{
    printf("(%d,%d,%d,%d)\n", size[0], size[1], size[2], size[3]);
}

template <typename T> void Tensor<T>::add(float coe, Tensor<T> *in)
{
    int i;
    for (i = 0; i < buff_size; i++) {
        buff[i] = buff[i] + coe * in->buff[i];
    }
}

template <typename T> void Tensor<T>::reload(Tensor<T> *in)
{
    memcpy(buff, in->buff, in->buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::disp()
{
    int i;
    for (i = 0; i < buff_size; i++) {
        cout << buff[i] << " ";
    }
    cout << endl;
}
#endif
