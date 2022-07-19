#include "SpeechWrap.h"
#include <stdio.h>
#include <string.h>

SpeechWrap::SpeechWrap()
{
    cache_size = 0;
}

SpeechWrap::~SpeechWrap()
{
}

void SpeechWrap::load(int16_t *din, int len)
{
    in = din;
    in_size = len;
    total_size = cache_size + in_size;
}

int SpeechWrap::size()
{
    return total_size;
}

void SpeechWrap::update(int offset)
{
    int in_offset = offset - cache_size;
    cache_size = (total_size - offset);
    memcpy(cache, in + in_offset, cache_size * sizeof(int16_t));
}

int16_t &SpeechWrap::operator[](int i)
{
    return i < cache_size ? cache[i] : in[i - cache_size];
}
