#include "fastasr.h"
#include "Audio.h"
#include "Model.h"

using namespace std;

void *model_create(struct ModelConfig config, int mode)
{

    Model *model = new Model(config, mode);
    return (void *)model;
}

void model_free(void *handle)
{
    Model *model = (Model *)handle;
    delete model;
}

void model_reset(void *handle)
{
    Model *model = (Model *)handle;
    model->reset();
}

char *model_forward_chunk(void *handle, short *din, int len, int flag)
{
    Model *model = (Model *)handle;
    string str = model->forward_chunk(din, len, flag);
    int ll = str.size();

    char *info = (char *)malloc(ll + 1);
    memcpy(info, str.c_str(), ll);
    info[ll] = 0;

    return info;
}

char *model_forward(void *handle, short *din, int len, int flag)
{
    Model *model = (Model *)handle;
    string str = model->forward(din, len, flag);
    int ll = str.size();

    char *info = (char *)malloc(ll + 1);
    memcpy(info, str.c_str(), ll);
    info[ll] = 0;

    return info;
}

char *model_rescoring(void *handle)
{
    Model *model = (Model *)handle;
    string str = model->rescoring();
    int ll = str.size();
    char *info = (char *)malloc(ll + 1);
    memcpy(info, str.c_str(), ll);
    info[ll] = 0;
    return info;
}

void *audio_create(int size)
{
    Audio *audio;
    if (size == 0)
        audio = new Audio();
    else
        audio = new Audio(size);

    return (void *)audio;
}

void audio_free(void *handle)
{

    Audio *audio = (Audio *)handle;
    delete audio;
}

void audio_disp(void *handle)
{

    Audio *audio = (Audio *)handle;
    audio->disp();
}

void audio_loadwav(void *handle, const char *filename)
{
    Audio *audio = (Audio *)handle;
    audio->loadwav(filename);
}

int audio_fetch_chunck(void *handle, int16_t **dout, int len)
{
    Audio *audio = (Audio *)handle;
    return (int)audio->fetch_chunck(*dout, len);
}

int audio_fetch(void *handle, int16_t **dout, int *len)
{
    Audio *audio = (Audio *)handle;
    return (int)audio->fetch(*dout, *len);
}
