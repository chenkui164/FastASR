#include <iostream>
#include <sys/time.h> // for gettimeofday()

#include "Audio.h"
#include "FeatureExtract.h"
#include "Model.h"

using namespace std;

int main(int argc, char *argv[])
{
    struct timeval start, end;
    Audio audio("ck.wav");
    FeatureExtract fe;
    ModelConfig cfg;
    cfg.vocab_path = "vocab.txt";
    cfg.wenet_path = "wenet_params.bin";

    Model mm(cfg);

    int16_t *buff;
    int len = 1360;
    int sum = 0;
    SpeechFlag flag = S_MIDDLE;
    while (flag == S_MIDDLE) {
        flag = audio.fetch_chunck(buff, len);
        sum += len;
        fe.insert(buff, len, flag);
    }

    int i = 0;
    int ll = fe.size();

    Tensor<float> *out;

    gettimeofday(&start, NULL);
    for (i = 0; i < ll; i++) {
        Tensor<float> *buff;
        fe.fetch(buff);
        string msg = mm.forward(buff, out);
        cout << msg << endl;
    }
    string msg = mm.rescoring();
    cout << msg << endl;
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs\n", (double)micros / 1000000);

    printf("fe size is %d\n", fe.size());
}
