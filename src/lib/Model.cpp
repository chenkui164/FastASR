#include "kaldi2/ModelImp.h"
#include "paddlespeech/ModelImp.h"
#include "paraformer/ModelImp.h"
#include <Model.h>

Model *create_model(const char *path, int mode)
{
    Model *mm;
    if (mode == 0 || mode == 1) {
        mm = new paddlespeech::ModelImp(path, mode);
    } else if (mode == 2) {
        mm = new kaldi2::ModelImp(path, mode);
    } else {
        mm = new paraformer::ModelImp(path, mode);
    }
    return mm;
}
