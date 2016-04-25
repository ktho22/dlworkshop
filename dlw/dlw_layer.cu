#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/dlw_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void DlwForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data, const Dtype* bias_data, const Dtype* slope2_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] > bias_data[c] ? in[index] * slope_data[c] : in[index] * slope2_data[c]; //change
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void DlwBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const Dtype* bias_data, const Dtype* slope2_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if ( in_data[index] > bias_data[c] ) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
    } else {
    out_diff[index] = in_diff[index] * slope2_data[c];  //change
    }
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void DlwParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* slope_diff, Dtype* bias_diff, Dtype* slope2_diff, const Dtype* bias_data, const int channels, const int dim) {    //change
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels;
    slope_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0) * (in_data[index] >= bias_data[c]);
    bias_diff[index] = 0;
    slope2_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= bias_data[c]);
    for ( int k = 1; k < rows; k++ ) {
        slope_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * (in_data[index + k*rowPitch] <= 0) * (in_data[index + k*rowPitch] >= bias_data[c]);
        slope2_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * (in_data[index + k*rowPitch] <= bias_data[c]);    //change
    }
  }
}

template <typename Dtype>
void DlwLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();
  const Dtype* slope2_data = this->blobs_[2]->gpu_data(); //attach
  const int div_factor = 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  DlwForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, bias_data, slope2_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void DlwLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    Dtype* slope2_diff = this->blobs_[2]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    DlwParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_slope_buff_.mutable_gpu_diff(), backward_bias_buff_.mutable_gpu_diff(), backward_slope2_buff_.mutable_gpu_diff(), this->blobs_[1]->gpu_data(), dim, channels);

    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_slope_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        slope_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_bias_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        bias_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_slope2_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        slope2_diff);
    
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    const Dtype* bias_data = this->blobs_[1]->gpu_data();
    const Dtype* slope2_data = this->blobs_[2]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    DlwBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data, bias_data, slope2_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DlwLayer);


}  // namespace caffe
