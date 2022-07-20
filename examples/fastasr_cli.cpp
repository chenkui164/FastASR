#include <iostream>
#include <sys/time.h>

#include "Audio.h"
#include "FeatureExtract.h"
#include "Model.h"

using namespace std;

int main(int argc, char *argv[])
{
    struct timeval start, end;
    Audio audio;
    audio.loadwav(argv[2]);
    audio.disp();

    gettimeofday(&start, NULL);
    Model mm(argv[1], 0);
    mm.reset();
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs\n", (double)micros / 1000000);

    int16_t *buff;
    int len;
    int flag = audio.fetch(buff, len);

    gettimeofday(&start, NULL);
    string msg = mm.forward(buff, len, flag);
    gettimeofday(&end, NULL);

    cout << "result: \"" << msg << "\"" << endl;

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    return 0;
}
