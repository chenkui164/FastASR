#include "FeatureQueue.h"
#include <string>

FeatureQueue::FeatureQueue()
{
    buff = new Tensor<float>(67, 80);
    buff_idx = 0;
}

FeatureQueue::~FeatureQueue()
{
}

void FeatureQueue::push(float *din, SpeechFlag flag)
{
    int offset = buff_idx * 80;
    memcpy(buff->buff + offset, din, 80 * sizeof(float));
    buff_idx++;

    if (buff_idx == 67) {
        feature_queue.push(buff);
        Tensor<float> *tmp = new Tensor<float>(67, 80);
        memcpy(tmp->buff, buff->buff + 64 * 80, 3 * 80 * sizeof(float));
        buff_idx = 3;
        buff = tmp;
    } else if (flag == S_END) {
        Tensor<float> *tmp = new Tensor<float>(buff_idx, 80);
        memcpy(tmp->buff, buff->buff, buff_idx * 80 * sizeof(float));
        feature_queue.push(tmp);
        buff_idx = 0;
    }
}

Tensor<float> *FeatureQueue::pop()
{

    Tensor<float> *tmp = feature_queue.front();
    feature_queue.pop();
    return tmp;
}

int FeatureQueue::size()
{
    return feature_queue.size();
}
