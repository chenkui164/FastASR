#ifndef FASTASR_H
#define FASTASR_H

#include "CommonStruct.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void *model_create(struct ModelConfig config, int mode);
extern void model_free(void *handle);
extern void model_reset(void *handle);
extern char *model_forward_chunk(void *handle, short *din, int len, int flag);
extern char *model_forward(void *handle, short *din, int len, int flag);
extern char *model_rescoring(void *handle);

extern void *audio_create(int size);
extern void audio_free(void *handle);
extern void audio_disp(void *handle);
extern void audio_loadwav(void *handle, const char *filename);
extern int audio_fetch_chunck(void *handle, int16_t **dout, int len);
extern int audio_fetch(void *handle, int16_t **dout, int *len);

#ifdef __cplusplus
}
#endif

#endif
