#include <iostream>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include <Audio.h>
#include <Model.h>

using namespace std;

int main(int argc, char *argv[])
{
    struct timeval start, end;
    Audio audio(1);
    audio.loadwav(argv[2]);
    audio.padding();
    audio.disp();

    gettimeofday(&start, NULL);
    Model *mm = create_model(argv[1], 2);
    mm->reset();
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)micros / 1000000);

    float *buff;
    int len;
    int flag = audio.fetch(buff, len);

    gettimeofday(&start, NULL);
    string msg = mm->forward(buff, len, flag);
    gettimeofday(&end, NULL);

    cout << "Result: \"" << msg << "\"." << endl;

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    return 0;
}
