#include "CTCDecode.h"
#include "util.h"
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string.h>
using namespace std;

#define vocab_size 5537

CTCdecode::CTCdecode(float *ctc_weight, float *ctc_bias)
    : ctc_weight(ctc_weight), ctc_bias(ctc_bias)
{
}

CTCdecode::~CTCdecode()
{
}

float log_add(float *din, int len)
{
    float sum = 0;
    int i;
    for (i = 0; i < len; i++) {
        sum = sum + exp(din[i]);
    }
    return log(sum);
}

auto char_cmp = [](CharProb a, CharProb b) { return a.prob < b.prob; };
auto path_cmp = [](PathProb a, PathProb b) { return a.prob < b.prob; };

void topk(float *din, int len, set<CharProb, decltype(char_cmp)> &s)
{
    int i;
    for (i = 0; i < 10; i++) {
        CharProb tmp;
        tmp.char_idx = i;
        tmp.prob = din[i];
        s.insert(tmp);
    }

    float min = s.begin()->prob;

    for (; i < len; i++) {
        if (din[i] > min) {
            s.erase(s.begin());
            CharProb tmp;
            tmp.char_idx = i;
            tmp.prob = din[i];
            s.insert(tmp);
            min = s.begin()->prob;
        }
    }
}

void ctc_beam_search(Tensor<float> *din, deque<PathProb> &hyps)
{
    int tmax = din->size[2];
    int beam_size = 10;
    int i;

    set<PathProb, decltype(path_cmp)> curr_hyps_set(path_cmp);
    PathProb tmp;
    tmp.pb = 0;
    tmp.prob = 0;
    curr_hyps_set.insert(tmp);

    for (i = 0; i < tmax; i++) {
        set<CharProb, decltype(char_cmp)> char_set(char_cmp);
        topk(din->buff + i * vocab_size, vocab_size, char_set);
        map<vector<int>, PathProb> next_next_map;
        for (auto char_it = char_set.begin(); char_it != char_set.end();
             ++char_it) {
            int char_idx = char_it->char_idx;
            float char_prob = char_it->prob;
            for (auto hyps_it = curr_hyps_set.begin();
                 hyps_it != curr_hyps_set.end(); hyps_it++) {
                int last = -1;
                if (hyps_it->prefix.size() > 0) {
                    int ii = hyps_it->prefix.size() - 1;
                    last = hyps_it->prefix[ii];
                }
                vector<int> curr_prefix(hyps_it->prefix);
                vector<int> next_prefix(hyps_it->prefix);
                next_prefix.push_back(char_idx);

                if (char_idx == 0) {
                    auto next_hyps = next_next_map[curr_prefix];
                    float tmp[] = {next_hyps.pb, hyps_it->pb + char_prob,
                                   hyps_it->pnb + char_prob};
                    next_hyps.pb = log_add(tmp, 3);
                    next_hyps.prefix = curr_prefix;
                    next_next_map[curr_prefix] = next_hyps;
                } else if (last == char_idx) {
                    {
                        auto next_hyps = next_next_map[curr_prefix];
                        float tmp[] = {next_hyps.pnb, hyps_it->pnb + char_prob};
                        next_hyps.pnb = log_add(tmp, 2);
                        next_hyps.prefix = curr_prefix;
                        next_next_map[curr_prefix] = next_hyps;
                    }

                    {
                        auto next_hyps = next_next_map[next_prefix];
                        float tmp[] = {next_hyps.pnb, hyps_it->pb + char_prob};
                        next_hyps.pnb = log_add(tmp, 2);
                        next_hyps.prefix = next_prefix;
                        next_next_map[next_prefix] = next_hyps;
                    }
                } else {
                    auto next_hyps = next_next_map[next_prefix];
                    float tmp[] = {next_hyps.pnb, hyps_it->pb + char_prob,
                                   hyps_it->pnb + char_prob};
                    next_hyps.pnb = log_add(tmp, 3);
                    next_hyps.prefix = next_prefix;
                    next_next_map[next_prefix] = next_hyps;
                }
            }
        }
        // kaishi
        float min = -9999999;
        int ii = 0;
        curr_hyps_set.clear();
        for (auto map_it = next_next_map.begin(); map_it != next_next_map.end();
             map_it++) {
            float tmp[] = {map_it->second.pb, map_it->second.pnb};
            map_it->second.prob = log_add(tmp, 2);
            if (ii < 10) {
                curr_hyps_set.insert(map_it->second);
                min = curr_hyps_set.begin()->prob;
                ii++;
            } else {
                if (min < map_it->second.prob) {
                    curr_hyps_set.insert(map_it->second);
                    curr_hyps_set.erase(curr_hyps_set.begin());
                    min = curr_hyps_set.begin()->prob;
                }
            }
        }
    }

    for (auto hyps_it = curr_hyps_set.begin(); hyps_it != curr_hyps_set.end();
         hyps_it++) {
        hyps.push_front(*hyps_it);
        // int mm = hyps_it->prefix.size();
        // cout << hyps_it->prefix.size() << endl;
        // for (i = 0; i < mm; i++) {
        //     printf("%d ", hyps_it->prefix[i]);
        // }
        // printf("\n");
        // printf("%f %f %f\n\n", hyps_it->pb, hyps_it->pnb, hyps_it->prob);
    }
}

void CTCdecode::show_hyps(deque<PathProb> hyps)
{
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int mm = hyps_it->prefix.size();
        int i;
        printf("prefix len is %d, val is [", mm);
        for (i = 0; i < mm - 1; i++) {
            printf("%d,", hyps_it->prefix[i]);
        }
        printf("%d]\n", hyps_it->prefix[i]);
        printf("pb is %f, pnb is %f, prob is %f\n", hyps_it->pb, hyps_it->pnb,
               hyps_it->prob);
    }
}

void CTCdecode::forward(Tensor<float> *din, deque<PathProb> &hyps)
{

    int mm = din->size[2];
    Tensor<float> ctcin(mm, vocab_size);
    int i;
    for (i = 0; i < mm; i++) {
        int offset = i * vocab_size;
        memcpy(ctcin.buff + offset, ctc_bias, sizeof(float) * vocab_size);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, vocab_size, 512,
                1, din->buff, 512, ctc_weight, vocab_size, 1, ctcin.buff,
                vocab_size);

    for (i = 0; i < mm; i++) {
        int offset = i * vocab_size;
        log_softmax(ctcin.buff + offset, vocab_size);
    }

    ctc_beam_search(&ctcin, hyps);
    // show_hyps(hyps);

}
