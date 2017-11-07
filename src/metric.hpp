/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: metric.hpp
 *     Created: 2017-11-07 16:45
 * Description:
 * ===================================================
 */
#ifndef _METRIC_HPP_
#define _METRIC_HPP_

// c++
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>
//
#include <common.hpp>

class Metric {
 public:
  bool init(NYUnpacker &unp,
          const std::string& pcameanFilePath,
          const std::string& evalsFilePath,
          const std::string& evecsFilePath);

  void process(float* fea, int dim, std::vector<float> &out,
          int pcadim=0, bool l2norm=false);

  void process(float* fea, int dim, std::vector<float> &out,
          double pcaratio=0, bool l2norm=false);

 private:
  cv::Mat pcamean_;
  cv::Mat evals_;
  cv::Mat evecs_;
  int dim_;
};

#endif  // _METRIC_HPP_
