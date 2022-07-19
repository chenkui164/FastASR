
#ifndef AUDIO_H
#define AUDIO_H

#include <fftw3.h>
#include <stdint.h>

#include "FeatureQueue.h"
#include "Tensor.h"

class Audio {
  private:
    int16_t *speech_data;
    int speech_len;
    int speech_align_len;
    int16_t sample_rate;
    int offset;

  public:
    Audio();
    ~Audio();
    void loadwav(const char *filename);
    SpeechFlag fetch_chunck(int16_t *&dout, int len);
    SpeechFlag fetch(int16_t *&dout, int &len);
};

#endif
