#ifndef CAFFE_CTC_DECODER_LAYER_HPP_
#define CAFFE_CTC_DECODER_LAYER_HPP_

#include <vector>
#include <string>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A layer that implements the decoder for a ctc
 *
 * Bottom blob is the probability of label and the sequence indicators.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CTCDecoderLayer : public Layer<Dtype> {
 public:
  typedef vector<int> Sequence;
  typedef vector<Sequence> Sequences;

 public:
  explicit CTCDecoderLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CTCDecoder"; }

  // probabilities (T x N x C),
  // sequence_indicators (T x N),
  // target_sequences (T X N) [optional]
  // if a target_sequence is provided, an additional accuracy top blob is
  // required
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // sequences (terminated with negative numbers),
  // output scores [optional if 2 top blobs and bottom blobs = 2]
  // accuracy [optional, if target_sequences as bottom blob = 3]
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  const Sequences& OutputSequences() const {return output_sequences_;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Best_Path_Decode(const Blob<Dtype>* probabilities,
                                const Blob<Dtype>* sequence_indicators,
                                Sequences* output_sequences,
                                Blob<Dtype>* scores);

  virtual void Prefix_Search_Decode(const Blob<Dtype>* probabilities,
                                    const Blob<Dtype>* sequence_indicators,
                                    Sequences* output_sequences,
                                    Blob<Dtype>* scores);

  int EditDistance(const Sequence &s1, const Sequence &s2);

  void recurionPath(const Blob<Dtype>* probabilities, int sample_n,
	                int section_end,
	                int position,
	                vector<pair<int, double> > pipath, //pi
	                map<string, double>& lpath);
  void mapipath(vector<pair<int, double> > pipath, string & key, double & value);
  vector<string> strsplit(string &str, const string &delim);

 protected:
  Sequences output_sequences_;
  int T_;
  int N_;
  int C_;
  int blank_index_;
  bool merge_repeated_;
  int decoding;
  float blank_threshold;
  string delim;

  int sequence_index_;
  int score_index_;
  int accuracy_index_;
};

}  // namespace caffe

#endif  // CAFFE_CTC_DECODER_LAYER_HPP_
