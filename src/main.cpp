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
    struct timeval start, end;

    Audio cc(argv[1]);

    gettimeofday(&start, NULL);

    Model mm;

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs\n", (double)micros / 1000000);

    Tensor<float> *dout;

    gettimeofday(&start, NULL);
    string result = mm.forward(cc.fbank_feature, dout);
    gettimeofday(&end, NULL);

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    cout << "result: \"" << result << "\"" << endl;

    printf("Model inference takes %lfs.\n", (double)micros / 1000000);
}
