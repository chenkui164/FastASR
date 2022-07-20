#include <iostream>
#include <sys/time.h> // for gettimeofday()

#include "Audio.h"
#include "FeatureExtract.h"
#include "Model.h"

using namespace std;

int main(int argc, char *argv[])
{
    struct timeval start, end;
    int len = 1360;
    Audio audio(len);
    ModelConfig cfg;

    cfg.vocab_path = "stream/vocab.txt";
    cfg.wenet_path = "stream/wenet_params.bin";
    Model mm(cfg, 1);

    int i;
    for (i = 0; i < 10; i++) {
        int16_t *buff;
        // int len = 1360;
        int sum = 0;
        audio.loadwav("zh.wav");
        mm.reset();

        int flag = S_MIDDLE;
        while (flag == S_MIDDLE) {
            flag = audio.fetch_chunck(buff, len);
            sum += len;
            string msg = mm.forward_chunk(buff, len, flag);
            cout << msg << endl;
        }

        string msg = mm.rescoring();
        cout << msg << endl;
    }
}
