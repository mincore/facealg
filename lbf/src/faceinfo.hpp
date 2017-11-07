/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: faceinfo.hpp
 *     Created: 2017-11-07 16:48
 * Description:
 * ===================================================
 */
#ifndef FACEINFO_HPP_
#define FACEINFO_HPP_

#include <vector>
#include <opencv2/opencv.hpp>

typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} FaceRect;

typedef struct FacePts {
  float x[5],y[5];
} FacePts;

typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
} FaceInfo;

typedef struct ImageInfo {
  cv::Mat image;
  int height;
  int width;
  std::vector<FaceInfo> faceInfo;
} ImageInfo;

#endif  // FACEINFO_HPP_
