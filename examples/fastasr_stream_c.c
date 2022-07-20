#include "fastasr.h"
#include <malloc.h>
#include <stdio.h>

int main()
{

    int len = 1360;
    void *audio_obj = audio_create(len);
    audio_loadwav(audio_obj, "zh.wav");
    audio_disp(audio_obj);


    void *modle_obj = model_create("./stream/", 1);
    model_reset(modle_obj);
    int flag = S_MIDDLE;
    int16_t *buff;

    while (flag == S_MIDDLE) {
        flag = audio_fetch_chunck(audio_obj, &buff, len);
        char *result = model_forward_chunk(modle_obj, buff, len, flag);
        printf("%s\n", result);
        free(result);
    }

    char *result = model_forward_chunk(modle_obj, buff, len, flag);
    printf("result :%s\n", result);

    audio_free(audio_obj);
    model_free(modle_obj);

    return 0;
}
