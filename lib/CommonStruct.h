
#ifndef COMMONSTRUCT_H
#define COMMONSTRUCT_H

struct ModelConfig {
    const char *vocab_path;
    const char *wenet_path;
};

#define S_BEGIN  0
#define S_MIDDLE 1
#define S_END    2
#define S_ALL    3
#define S_ERR    4

#endif
