#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/dlw_layer.hpp"

namespace caffe {

template <typename Dtype>
void DlwLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  DlwParameter dlw_param = this->layer_param().dlw_param();
  int channels = bottom[0]->channels();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);     // We have two parameters. so change 1 -> 2
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
    this->blobs_[2].reset(new Blob<Dtype>(vector<int>(1, channels))); // initialize the size of slope and bias

    shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
		this->layer_param_.dlw_param().slope_filler()));   //slope filler "constant" and initial slope value
    slope_filler->Fill(this->blobs_[0].get()); //slope parameter initialization

    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
		this->layer_param_.dlw_param().bias_filler()));  //bias filler "constant" and initial bias value
    bias_filler->Fill(this->blobs_[1].get()); //bias parameter initialization

    shared_ptr<Filler<Dtype> > slope2_filler(GetFiller<Dtype>(
		this->layer_param_.dlw_param().slope2_filler()));   //slope2 filler "constant" and initial slope2 value
    slope2_filler->Fill(this->blobs_[0].get()); //slope2 parameter initialization

  }

  CHECK_EQ(this->blobs_[0]->count(), channels)
      << "Negative slope size is inconsistent with prototxt config";
  CHECK_EQ(this->blobs_[1]->count(), channels)
      << "bias size is inconsistent with prototxt config";
  CHECK_EQ(this->blobs_[2]->count(), channels)
      << "Negative slope2 size is inconsistent with prototxt config";
  

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_slope_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_slope2_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_bias_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));    // add bias buff. 
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void DlwLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void DlwLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->cpu_data(); // slope
  const Dtype* bias_data = this->blobs_[1]->cpu_data();  // bias
  const Dtype* slope2_data = this->blobs_[2]->cpu_data(); // slope2

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = 1; // we do not share the weight along the channel
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    if (bottom_data[i] > bias_data[c]) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + slope_data[c] * std::min(bottom_data[i], Dtype(0));
    } else { 
    top_data[i] = slope2_data[c] * bottom_data[i];
    }   //change
  }
}

template <typename Dtype>
void DlwLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* slope2_data = this->blobs_[2]->cpu_data();
  const Dtype* bias_data = this->blobs_[1]->cpu_data(); //add
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = 1; // we do not share the weight along the channel

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) { // parameter update
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    Dtype* slope2_diff = this->blobs_[2]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      slope_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0) * (bottom_data[i] >= bias_data[c]); //change
      bias_diff[c] += 0; //change
      slope2_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= bias_data[c]); //change
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      if (bottom_data[i] > bias_data[c]) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
      } else {
      bottom_diff[i] = top_diff[i] * slope2_data[c];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DlwLayer);
#endif

INSTANTIATE_CLASS(DlwLayer);
REGISTER_LAYER_CLASS(Dlw);

}  // namespace caffe
