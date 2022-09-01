
#ifndef AUDIO_H
#define AUDIO_H

#include <stdint.h>
#include <ComDefine.h>


class Audio {
  private:
    float *speech_data;
    int16_t *speech_buff;
    int speech_len;
    int speech_align_len;
    int16_t sample_rate;
    int offset;
    float align_size;
    int data_type;

  public:
    Audio(int data_type);
    Audio(int data_type, int size);
    ~Audio();
    void disp();
    void loadwav(const char *filename);
    int fetch_chunck(float *&dout, int len);
    int fetch(float *&dout, int &len);
    void padding();
};

#endif
