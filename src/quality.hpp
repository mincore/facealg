/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: quality.hpp
 *     Created: 2017-11-07 16:46
 * Description:
 * ===================================================
 */
#ifndef _QUALITY_HPP_
#define _QUALITY_HPP_

// caffe
#include <caffe/caffe.hpp>
// c++
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>
//
#include <faceinfo.hpp>
#include <common.hpp>

using namespace caffe;

class Quality {
 public:
  bool init(NYUnpacker &unp,
          const std::string &modelFilePath,
          const std::string &trainedFilePath,
          int device=0);
  void prealloc(int times=1);

  float detect(const cv::Mat& img);

 private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  cv::Mat Preprocess(const cv::Mat& img);

 private:
  bool is_inited_;
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  double bias_;
  double scale_;
};

#endif  // _QUALITY_HPP_
