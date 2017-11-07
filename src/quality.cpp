/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: quality.cpp
 *     Created: 2017-11-07 16:46
 * Description:
 * ===================================================
 */
// caffe
#include <caffe/caffe.hpp>
// c++
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>

#include "load.h"

bool Quality::init(NYUnpacker &unp,
                const std::string &modelFilePath,
                const std::string &trainedFilePath,
                int device) {
  Caffe::SetDevice(device);
  Caffe::set_mode(Caffe::GPU);
  //std::cout << "Using GPU" << std::endl;

  /* Load the network. */
  //net_.reset(new Net<float>(modelFilePath, TEST));
  //net_->CopyTrainedLayersFrom(trainedFilePath);
  if (!LoadNet(unp, net_, modelFilePath, trainedFilePath)) {
      printf("load net failed\n");
      return false;
  }

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  bias_ = 0;
  scale_ = 0.00390625;

  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  // Forward dimension change to all layers.
  net_->Reshape();

  is_inited_ = true;
  return is_inited_;
}

void Quality::prealloc(int times) {
  if (!is_inited_) {
    std::cout << "Quality Not Inited." << std::endl;
    return;
  }
  cv::Mat t;
  t = cv::Mat::zeros(input_geometry_, CV_32FC3);
  for (int i = 0; i < times; ++i) {
    detect(t);
  }
}

float Quality::detect(const cv::Mat& img) {
  if (!is_inited_) {
    std::cout << "Quality Not Inited." << std::endl;
    return 0;
  }

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

#ifndef NDEBUG
  double t1 = getTime();
#endif
  cv::Mat sample_final;
  sample_final = Preprocess(img);
#ifndef NDEBUG
  double t2 = getTime();
  std::cout << "Time of Quality PREPROCESS: " << t2-t1 << std::endl;
#endif

  cv::split(sample_final, input_channels);

  CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

  net_->Forward();
#ifndef NDEBUG
  double t3 = getTime();
  std::cout << "Time of Quality FORWARD: " << t3-t2 << std::endl;
#endif

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  float score = output_layer->cpu_data()[0];
  return score;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Quality::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

cv::Mat Quality::Preprocess(const cv::Mat& img) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_) {
    cv::resize(sample, sample_resized, input_geometry_);
  } else {
    sample_resized = sample;
  }

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3, scale_, -scale_*bias_);
  else
    sample_resized.convertTo(sample_float, CV_32FC1, scale_, -scale_*bias_);

  return sample_float;
}
