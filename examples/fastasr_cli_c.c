#include "fastasr.h"
#include <malloc.h>
#include <stdio.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{

    struct timeval start, end;

    int16_t *buff;
    int len;
    void *audio_obj = audio_create(0);
    audio_loadwav(audio_obj, argv[2]);
    audio_disp(audio_obj);
    int flag = audio_fetch(audio_obj, &buff, &len);

    gettimeofday(&start, NULL);
    void *modle_obj = model_create(argv[1], 0);
    model_reset(modle_obj);

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs\n", (double)micros / 1000000);

    gettimeofday(&start, NULL);
    char *result = model_forward(modle_obj, buff, len, flag);
    gettimeofday(&end, NULL);
    printf("result: \"%s\"\n", result);

    audio_free(audio_obj);
    model_free(modle_obj);

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    free(result);

    return 0;
}
