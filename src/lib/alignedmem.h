
#ifndef ALIGNEDMEM_H
#define ALIGNEDMEM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *aligned_malloc(size_t alignment, size_t required_bytes);
extern void aligned_free(void *p);

#endif
