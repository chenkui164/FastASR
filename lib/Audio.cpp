#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Audio.h"

using namespace std;

Audio::Audio(int data_type) : data_type(data_type)
{
    speech_buff = NULL;
    speech_data = NULL;
    align_size = 1360;
}

Audio::Audio(int data_type, int size) : data_type(data_type)
{
    speech_buff = NULL;
    speech_data = NULL;
    align_size = (float)size;
}

Audio::~Audio()
{
    if (speech_buff != NULL) {
        free(speech_buff);
        free(speech_data);
    }
}

void Audio::disp()
{
    printf("Audio time is %f s. len is %d\n", (float)speech_len / 16000,
           speech_len);
}

void Audio::loadwav(const char *filename)
{

    if (speech_buff != NULL) {
        free(speech_buff);
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
    speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_align_len);
    memset(speech_buff, 0, sizeof(int16_t) * speech_align_len);
    int ret = fread(speech_buff, sizeof(int16_t), speech_len, fp);
    fclose(fp);

    speech_data = (float *)malloc(sizeof(float) * speech_align_len);
    memset(speech_data, 0, sizeof(float) * speech_align_len);
    int i;
    float scale = 1;

    if (data_type == 1) {
        scale = 32768;
    }

    for (i = 0; i < speech_len; i++) {
        speech_data[i] = (float)speech_buff[i] / scale;
    }
}

int Audio::fetch_chunck(float *&dout, int len)
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

int Audio::fetch(float *&dout, int &len)
{
    dout = speech_data;
    len = speech_len;
    offset = speech_len;
    return S_END;
}

void Audio::padding()
{

    float num_samples = speech_len;
    float frame_length = 400;
    float frame_shift = 160;
    float num_frames = floor((num_samples + (frame_shift / 2)) / frame_shift);
    float num_new_samples = (num_frames - 1) * frame_shift + frame_length;
    float num_padding = num_new_samples - num_samples;
    float num_left_padding = (frame_length - frame_shift) / 2;
    float num_right_padding = num_padding - num_left_padding;

    float *new_data = (float *)malloc(num_new_samples * sizeof(float));
    int i;
    int tmp_off = 0;
    for (i = 0; i < num_left_padding; i++) {
        int ii = num_left_padding - i - 1;
        new_data[i] = speech_data[ii];
    }
    tmp_off = num_left_padding;
    memcpy(new_data + tmp_off, speech_data, speech_len * sizeof(float));
    tmp_off += speech_len;

    for (i = 0; i < num_right_padding; i++) {
        int ii = speech_len - i - 1;
        new_data[tmp_off + i] = speech_data[ii];
    }
    free(speech_data);
    speech_data = new_data;
    speech_len = num_new_samples;

    // printf("num_new_samples is %f\n", num_new_samples);
    // printf("num_left_padding is %f\n", num_left_padding);
    // printf("num_right_padding is %f\n", num_right_padding);
}
