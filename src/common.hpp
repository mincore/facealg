/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: common.hpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <faceinfo.hpp>
#include "nypack.h"

enum cmpType{SCORE, AREA, CENTER};
class cmpFunc
{
  public:
    cmpFunc(cmpType c=SCORE, int height=0, int width=0);
    bool operator () (const FaceInfo& a, const FaceInfo& b);
  private:
    cmpType cmptype;
    float h, w;
};

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Minimum)
enum nmsType{UNION, MINIMUM};
std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
                float thresh, nmsType nmstype);

cv::Mat PCA(const cv::Mat& data,
        const cv::Mat& mean, const cv::Mat& evecs, int pcadim=0);

float OneOne(const cv::Mat& feam1, const cv::Mat& feam2,
        bool l2norm=true);
float OneOne(float* fea1, float* fea2,
        bool l2norm=true);

template <typename T>
cv::Mat Pointer2Mat(T* ptr, int height, int width, int type=CV_8UC3);

void drawFaceInfo(cv::Mat& img, const FaceInfo& faceInfo, bool drawPTs=true);
void drawFaceInfo(cv::Mat& img, const std::vector<FaceInfo>& faceInfos, bool drawPTs=true);

int txt2Mat(const std::string& txtFilePath, cv::Mat& mat,
        int rows=0, int cols=0, int c=0);

double getTime();

#endif  // _COMMON_HPP_
