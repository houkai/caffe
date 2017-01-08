#include "caffe/layers/ctc_decoder_layer.hpp"

#include <algorithm>
#include <vector>
#include <map>
#include <sstream>

// Base decoder
// ============================================================================

namespace caffe {
    template <typename Dtype>
        CTCDecoderLayer<Dtype>::CTCDecoderLayer(const LayerParameter& param)
        : Layer<Dtype>(param)
          , T_(0)
          , N_(0)
          , C_(0)
          , blank_index_(param.ctc_decoder_param().blank_index())
          , merge_repeated_(param.ctc_decoder_param().ctc_merge_repeated())
          , sequence_index_(0)
          , score_index_(-1)
          , accuracy_index_(-1) {
              delim = "_";
              decoding = param.ctc_decoder_param().decoding(); // decode method
              CHECK(decoding == 0 || decoding == 1) << "decode method must be 0 or 1";
              if(decoding == 1){
                  blank_threshold = param.ctc_decoder_param().blank_threshold();
              }
          }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            // compute indices of output (top) blobs
            sequence_index_ = 0;  // always
            if (bottom.size() == 2) {
                // 2 input blobs (data, sequence indicators)

                if (top.size() == 1) {
                    // no further output
                } else if (top.size() == 2) {
                    score_index_ = 1;  // output scores
                } else {
                    LOG(FATAL) << "Only two output blobs allowed: "
                        << "1: sequences, 2: scores";
                }
            } else if (bottom.size() == 3) {
                // 3 input blobs (data, seq_ind, target_seq)
                if (top.size() == 1) {
                    // no further output
                } else if (top.size() == 2) {
                    accuracy_index_ = 1;  // output accuracy
                } else if (top.size() == 3) {
                    score_index_ = 1;  // output scores
                    accuracy_index_ = 2;  // output accuracy
                } else {
                    LOG(FATAL) << "Need two or three output blobs: "
                        << "a) 1: sequences, 2: accuracy, or "
                        << "b) 1: sequences, 2: score, 3: accuracy.";
                }
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            const Blob<Dtype>* probabilities = bottom[0];
            T_ = probabilities->shape(0);
            N_ = probabilities->shape(1);
            C_ = probabilities->shape(2);

            output_sequences_.clear();
            output_sequences_.resize(N_);

            if (sequence_index_ >= 0) {
                Blob<Dtype>* sequences = top[sequence_index_];
                sequences->Reshape(N_, T_, 1, 1);  // switch to N x T
            }

            if (score_index_ >= 0) {
                Blob<Dtype>* scores = top[score_index_];
                scores->Reshape(N_, 1, 1, 1);
            }

            if (accuracy_index_ >= 0) {
                Blob<Dtype>* accuracy = top[accuracy_index_];
                accuracy->Reshape(1, 1, 1, 1);
            }

            if (blank_index_ < 0) {
                blank_index_ = C_ - 1;
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            const Blob<Dtype>* probabilities = bottom[0];
            const Blob<Dtype>* sequence_indicators = bottom[1];
            Blob<Dtype>* scores = (score_index_ >= 0) ? top[score_index_] : 0;

            // decode string with the requiested method
            if(decoding == 0) {
                Best_Path_Decode(probabilities, sequence_indicators, &output_sequences_, scores);
            } else {
                Prefix_Search_Decode(probabilities, sequence_indicators, &output_sequences_, scores);
            }

            // transform output_sequences to blob
            if (sequence_index_ >= 0) {
                Blob<Dtype>* sequence_blob = top[sequence_index_];
                Dtype* sequence_d = sequence_blob->mutable_cpu_data();
                // clear all data
                caffe_set(sequence_blob->count(), static_cast<Dtype>(-1), sequence_d);

                // copy data
                for (int n = 0; n < N_; ++n) {
                    Dtype* seq_n_d = sequence_d + sequence_blob->offset(n, 0);
                    const Sequence &output_seq = output_sequences_[n];
                    CHECK_LE(output_seq.size(), T_);
                    for (size_t t = 0; t < output_seq.size(); ++t) {
                        seq_n_d[t] = output_seq[t];
                    }
                }
            }

            // compute accuracy
            if (accuracy_index_ >= 0) {
                Dtype &acc = top[accuracy_index_]->mutable_cpu_data()[0];
                acc = 0;

                CHECK_GE(bottom.size(), 3);  // required target sequences blob
                const Blob<Dtype>* target_sequences_data = bottom[2];
                const Dtype* ts_data = target_sequences_data->cpu_data();
                for (int n = 0; n < N_; ++n) {
                    Sequence target_sequence;
                    for (int t = 0; t < T_; ++t) {
                        const Dtype dtarget = ts_data[target_sequences_data->offset(t, n)];
                        if (dtarget < 0) {
                            // sequence has finished
                            break;
                        }
                        // round to int, just to be sure
                        const int target = static_cast<int>(0.5 + dtarget);
                        target_sequence.push_back(target);
                    }

                    if (std::max(target_sequence.size(), output_sequences_[n].size()) == 0) {
                        // 0 length
                        continue;
                    }

                    const int ed = EditDistance(target_sequence, output_sequences_[n]);

                    acc += ed * 1.0 /
                        std::max(target_sequence.size(), output_sequences_[n].size());
                }

                acc = 1 - acc / N_;
                CHECK_GE(acc, 0);
                CHECK_LE(acc, 1);
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom) {
            for (int i = 0; i < propagate_down.size(); ++i) {
                if (propagate_down[i]) { NOT_IMPLEMENTED; }
            }
        }

    template <typename Dtype>
        int CTCDecoderLayer<Dtype>::EditDistance(const Sequence &s1,
                const Sequence &s2) {
            const size_t len1 = s1.size();
            const size_t len2 = s2.size();

            Sequences d(len1 + 1, Sequence(len2 + 1));

            d[0][0] = 0;
            for (size_t i = 1; i <= len1; ++i) {d[i][0] = i;}
            for (size_t i = 1; i <= len2; ++i) {d[0][i] = i;}

            for (size_t i = 1; i <= len1; ++i) {
                for (size_t j = 1; j <= len2; ++j) {
                    d[i][j] = std::min(
                            std::min(
                                d[i - 1][j] + 1,
                                d[i][j - 1] + 1),
                            d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1));
                }
            }

            return d[len1][len2];
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::Best_Path_Decode(
                const Blob<Dtype>* probabilities,
                const Blob<Dtype>* sequence_indicators,
                Sequences* output_sequences,
                Blob<Dtype>* scores) {
            Dtype* score_data = 0;
            if (scores) {
                CHECK_EQ(scores->count(), N_);
                score_data = scores->mutable_cpu_data();
                caffe_set(N_, static_cast<Dtype>(1.0), score_data);
            }

            for (int n = 0; n < N_; ++n) {
                int prev_class_idx = -1;

                for (int t = 0; /* check at end */; ++t) {
                    // get maximum probability and its index
                    int max_class_idx = 0;
                    const Dtype* probs = probabilities->cpu_data()
                        + probabilities->offset(t, n);
                    Dtype max_prob = probs[0];
                    ++probs;
                    for (int c = 1; c < C_; ++c, ++probs) {
                        if (*probs > max_prob) {
                            max_class_idx = c;
                            max_prob = *probs;
                        }
                    }

                    if (score_data) {
                        score_data[n] *= max_prob;
                    }

                    if (max_class_idx != blank_index_
                            && !(merge_repeated_&& max_class_idx == prev_class_idx)) {
                        output_sequences->at(n).push_back(max_class_idx);
                    }

                    prev_class_idx = max_class_idx;

                    if (t + 1 == T_ || sequence_indicators->data_at(t + 1, n, 0, 0) == 0) {
                        // End of sequence
                        break;
                    }
                }
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::Prefix_Search_Decode(
                const Blob<Dtype>* probabilities,
                const Blob<Dtype>* sequence_indicators,
                Sequences* output_sequences,
                Blob<Dtype>* scores) {
            Dtype* score_data = 0;
            if (scores) {
                CHECK_EQ(scores->count(), N_);
                score_data = scores->mutable_cpu_data();
                caffe_set(N_, static_cast<Dtype>(1.0), score_data);
            }

            for (int n = 0; n < N_; ++n) {
                //process one sample
                Sequences sections;
                Sequence section;
                //divide sequence into sections that are very likely to begin and end with a blank
                for (int t=0; /* check at end */; ++t) {
                    const Dtype* probs = probabilities->cpu_data()
                        + probabilities->offset(t, n);
                    //the prob of blank in t
                    Dtype blank_prob = *(probs + blank_index_);
                    if(blank_prob > blank_threshold){
                        if( !section.empty() ){
                            sections.push_back(section);
                            section.clear();
                        }	
                    } else {
                        section.push_back(t);
                    }
                    if (t + 1 == T_ || sequence_indicators->data_at(t + 1, n, 0, 0) == 0) {
                        // End of sequence
                        if(!section.empty()) {
                            sections.push_back(section);
                            section.clear();
                        }
                        break;
                    }
                }
                //calculate the most probable labelling for each section individually
                // and concatenate these to get the final classification
                ostringstream ostr;
                for(int i=0; i< sections.size(); i++) {
                    Sequence section = sections[i];
                    map<string, double> lpath;
                    recurionPath(
                            probabilities, n,
                            section.back()+1,
                            section[0],
                            vector<pair<int, double> >(),
                            lpath
                            );
                    if(lpath.empty()){
                        continue;
                    }
                    string maxkey;//is null string or string divided by delim
                    double maxprob = 0.;
                    for(map<string, double>::iterator it = lpath.begin(); it!=lpath.end(); it++) {
                        double prob_value = it->second;
                        string key = it->first;
                        if(prob_value > maxprob) {
                            maxprob = prob_value;
                            maxkey = key;
                        }
                    }
                    ostr << maxkey;
                    if(score_data) {
                        score_data[n] *= maxprob;
                    }
                }
                string outstring = ostr.str();
                vector<string> class_idxes = strsplit(outstring, delim);
                for(size_t i=0; i< class_idxes.size(); i++) {
                    int class_idx = atoi(class_idxes[i].c_str());
                    output_sequences->at(n).push_back(class_idx);
                }
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::recurionPath(
                const Blob<Dtype>* probabilities, int sample_n,
                int section_end,
                int position,
                vector<pair<int, double> > pipath, //pi
                map<string, double>& lpath
                ) {
            if(position >= section_end) {
                if(pipath.empty()) {
                    return;
                }
                string key; 
                double prob_value;
                mapipath(pipath, key, prob_value);//key maybe null string
                if(lpath.find(key) == lpath.end()){
                    lpath[key] = 0.0;
                }
                lpath[key] += prob_value;
                return;
            }
            //label
            const Dtype* probs = probabilities->cpu_data()
                + probabilities->offset(position, sample_n);
            for(int i=0; i< C_; i++) {
                pipath.push_back(pair<int, double>(i, *(probs+i)));
                recurionPath(probabilities, sample_n, section_end, position+1, pipath, lpath);
                pipath.pop_back();
            }
        }

    template <typename Dtype>
        void CTCDecoderLayer<Dtype>::mapipath(vector<pair<int, double> > pipath, string & key, double & value) {
            ostringstream ostr;
            value = 1.0;
            int prev_class_idx = blank_index_;
            for(size_t i=0; i< pipath.size(); i++) {
                int class_idx = pipath[i].first;
                double prob = pipath[i].second;
                value *= prob;
                if(class_idx != blank_index_ &&
                        !(merge_repeated_ && class_idx == prev_class_idx)) {
                    ostr <<  class_idx << delim;
                }
                prev_class_idx = class_idx;
            }
            key = ostr.str();
        }

    template <typename Dtype>
        vector<string> CTCDecoderLayer<Dtype>::strsplit(string &str, const string &delim)
        {
            vector<string> result;
            size_t last = 0;
            size_t index = str.find_first_of(delim, last);
            while(index != string::npos)
            {
                if(index - last >0)
                {
                    result.push_back(str.substr(last, index-last));
                }
                last = index+1;
                index = str.find_first_of(delim, last);
            }
            if(last < str.size())
            {
                result.push_back(str.substr(last, index-last));
            }
            return result;
        }


    INSTANTIATE_CLASS(CTCDecoderLayer);
    REGISTER_LAYER_CLASS(CTCDecoder);

}  // namespace caffe
