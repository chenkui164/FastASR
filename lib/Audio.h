
#ifndef AUDIO_H
#define AUDIO_H

#include <fftw3.h>
#include <stdint.h>

#include "FeatureQueue.h"
#include "Tensor.h"
#include "CommonStruct.h"

class Audio {
  private:
    int16_t *speech_data;
    int speech_len;
    int speech_align_len;
    int16_t sample_rate;
    int offset;
    float align_size;

  public:
    Audio(int size);
    Audio();
    ~Audio();
    void disp();
    void loadwav(const char *filename);
    int fetch_chunck(int16_t *&dout, int len);
    int fetch(int16_t *&dout, int &len);
};

#endif
