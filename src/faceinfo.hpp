/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: faceinfo.hpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
#ifndef _FACEINFO_HPP_
#define _FACEINFO_HPP_

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
  float quality;
  float age;
  float yaw;
  float pitch;
  float sex;
} FaceInfo;

#endif  // _FACEINFO_HPP_
