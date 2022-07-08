#include <iostream>
#include <malloc.h>
#include <map>
#include <set>
#include <stdint.h>
#include <string>
#include <sys/time.h> // for gettimeofday()

#include "Audio.h"
#include "Model.h"
#include <math.h>
// #include "WenetParams.h"
#include "Tensor.h"
#include "Vocab.h"
#include "util.h"

using namespace std;

int main(int argc, char *argv[])
{
    float *audio_in;
    Audio cc(argv[1]);
    Model mm;
    Tensor<float> *dout;

    struct timeval start, end;
    gettimeofday(&start, NULL);
    string result = mm.forward(cc.fbank_feature, dout);
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    cout << "result: \"" << result << "\"" << endl;

    printf("inference time is %lf s.\n", (double)micros / 1000000);
}
