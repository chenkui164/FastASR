#include "paddlespeech/ModelImp.h"
#include "kaldi2/ModelImp.h"
#include <Model.h>

Model *create_model(const char *path, int mode)
{
    // Model *mm = new paddlespeech::ModelImp(path, mode);
    Model *mm = new kaldi2::ModelImp(path, mode);
    return mm;
}
