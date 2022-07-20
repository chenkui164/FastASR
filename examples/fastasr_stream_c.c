#include <malloc.h>
#include <stdio.h>
#include <sys/time.h>

#include "fastasr.h"

int main(int argc, char *argv[])
{

    struct timeval start, end;
    int len = 1360;
    void *audio_obj = audio_create(len);
    audio_loadwav(audio_obj, argv[2]);
    audio_disp(audio_obj);

    gettimeofday(&start, NULL);
    void *modle_obj = model_create(argv[1], 1);
    model_reset(modle_obj);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs\n", (double)micros / 1000000);

    int flag = S_MIDDLE;
    int16_t *buff;

    gettimeofday(&start, NULL);
    while (flag == S_MIDDLE) {
        flag = audio_fetch_chunck(audio_obj, &buff, len);
        char *result = model_forward_chunk(modle_obj, buff, len, flag);
        printf("current result: \"%s\"\n", result);
        free(result);
    }

    char *result = model_rescoring(modle_obj);
    gettimeofday(&end, NULL);
    printf("final result: \"%s\"\n", result);

    audio_free(audio_obj);
    model_free(modle_obj);
    free(result);

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model inference takes %lfs.\n", (double)micros / 1000000);

    return 0;
}
