
#ifndef SPEECHWRAP_H
#define SPEECHWRAP_H

#include <stdint.h>

class SpeechWrap {
  private:
    int16_t cache[400];
    int cache_size;
    int16_t *in;
    int in_size;
    int total_size;
    int next_cache_size;

  public:
    SpeechWrap();
    ~SpeechWrap();
    void load(int16_t *din, int len);
    void update(int offset);
    void reset();
    int size();
    int16_t &operator[](int i);
};

#endif
