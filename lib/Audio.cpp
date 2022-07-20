#include "Audio.h"
#include "predefine_coe.h"
#include <fftw3.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
using namespace std;

Audio::Audio(int size)
{
    speech_data = NULL;
    align_size = (float)size;
}

Audio::Audio()
{
    speech_data = NULL;
    align_size = 1360;
}

Audio::~Audio()
{
    if (speech_data != NULL) {
        free(speech_data);
    }
}

void Audio::disp()
{
    printf("Audio time is %f s.\n", (float)speech_len / 16000);
}

void Audio::loadwav(const char *filename)
{

    if (speech_data != NULL) {
        free(speech_data);
    }
    offset = 0;

    FILE *fp;
    fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    uint32_t nFileLen = ftell(fp);
    fseek(fp, 44, SEEK_SET);

    speech_len = (nFileLen - 44) / 2;
    speech_align_len = (int)(ceil((float)speech_len / align_size) * align_size);
    speech_data = (int16_t *)malloc(sizeof(int16_t) * speech_align_len);
    memset(speech_data, 0, sizeof(int16_t) * speech_align_len);
    int ret = fread(speech_data, sizeof(int16_t), speech_len, fp);
    fclose(fp);
}

int Audio::fetch_chunck(int16_t *&dout, int len)
{
    if (offset >= speech_align_len) {
        dout = NULL;
        return S_ERR;
    } else if (offset == speech_align_len - len) {
        dout = speech_data + offset;
        offset = speech_align_len;
        return S_END;
    } else {
        dout = speech_data + offset;
        offset += len;
        return S_MIDDLE;
    }
}

int Audio::fetch(int16_t *&dout, int &len)
{
    dout = speech_data;
    len = speech_len;
    offset = speech_len;
    return S_END;
}
