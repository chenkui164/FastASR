#include "WenetParams.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// void ModelParams::load(const char *filename)
// {

//     FILE *fp;
//     fp = fopen(filename, "rb");
//     fseek(fp, 0, SEEK_END);
//     uint32_t nFileLen = ftell(fp);
//     fseek(fp, 0, SEEK_SET);
//     printf("nFileLen is %d\n", nFileLen);

//     params = (WenetParams *)aligned_malloc(32, nFileLen);
//     int n = fread(params, 1, nFileLen, fp);
//     printf("n is %d\n", n);

//     fclose(fp);
// }

// ModelParams::ModelParams(const char *filename)
// {
//     load(filename);
//     int i = 0;
//     for (i = 0; i < 10; i++) {
//         float a = params->encoder.embed.conv0_bias[i];
//         printf("%f ", a);
//     }
//     printf("\n");
// }

// ModelParams::~ModelParams()
// {
// }
