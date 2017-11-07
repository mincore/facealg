/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: property.cpp
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

#include "property.hpp"
#include "load.h"

bool Property::init(NYUnpacker &unp,
                 const std::string& modelFilePath,
                 const std::string& trainedFilePath,
                 int device) {
  Caffe::SetDevice(device);
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  //net_.reset(new Net<float>(modelFilePath, TEST));
  //net_->CopyTrainedLayersFrom(trainedFilePath);
  if (!LoadNet(unp, net_, modelFilePath, trainedFilePath)) {
      printf("load net failed\n");
      return false;
  }

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 3) << "Network should have exactly one output.";
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

void Property::prealloc(int times) {
  if (!is_inited_) {
    std::cout << "Property Not Inited." << std::endl;
    return;
  }
  cv::Mat t;
  t = cv::Mat::zeros(input_geometry_, CV_32FC3);
  for (int i = 0; i < times; ++i) {
    detect(t);
  }
}
vector<float> Property::detect(const cv::Mat& img) {
  if (!is_inited_) {
     vector<float> temp;
    std::cout << "Property Not Inited." << std::endl;
    return temp;
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
  std::cout << "Time of Property PREPROCESS: " << t2-t1 << std::endl;
#endif
  cv::split(sample_final, input_channels);
  CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
  net_->Forward();
#ifndef NDEBUG
  double t3 = getTime();
  std::cout << "Time of Property FORWARD: " << t3-t2 << std::endl;
#endif
  /* Copy the output layer to a std::vector */
  vector<Blob<float>*> output_layer = net_->output_blobs();
  //std::cout<<output_layer<<std::endl;
  float age = output_layer[0]->cpu_data()[0];
  float yaw   = output_layer[1]->cpu_data()[0];
  float pitch = output_layer[1]->cpu_data()[1];
  float sex =   output_layer[2]->cpu_data()[0];
  //std::cout<<"***"<<age<<" "<<yaw<<" "<<pitch<<" "<<sex<<std::endl;
  vector<float> result ;
  result.push_back(age);
  result.push_back(yaw);
  result.push_back(pitch);
  result.push_back(sex);

  return result;
}
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * operation will write the separate channels directly to the input
 * layer. */
void Property::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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
cv::Mat Property::Preprocess(const cv::Mat& img) {
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

#if 0
int main(int argc, char **argv){
        string name = argv[1];
        string modelFilePath="/home/hyh/caffe/examples/train_property/deploy.prototxt";
        string trainedFilePath = "/home/hyh/classify/_iter_50000.caffemodel";
        Property temp(modelFilePath , trainedFilePath);
        cv::Mat img = cv::imread(name);
        vector<float> score = temp.detect(img);
        for (int i;i<score.size();i++)
        std::cout<<score[i]<<std::endl;

        return 0;
        }
#endif
