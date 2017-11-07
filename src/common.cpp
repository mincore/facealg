/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: common.cpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
#include <sys/time.h>
#include <fstream>
#include <iterator>
#include <faceinfo.hpp>
#include <common.hpp>

cmpFunc::cmpFunc(cmpType c, int height, int width):cmptype(c),h(height),w(width) {}
bool cmpFunc::operator () (const FaceInfo& a, const FaceInfo& b) {
    float area1, area2;
    float center_x, center_y;
    float center_x1, center_y1;
    float center_x2, center_y2;
    float distance_1, distance_2;
    switch(cmptype) {
    case SCORE:
        return a.bbox.score > b.bbox.score;
    case AREA:
        area1 = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
        area2 = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
        return area1 > area2;
    case CENTER:
        center_x = w / 2;
        center_y = h / 2;
        center_x1 = (a.bbox.x2 + a.bbox.x1)/2;
        center_y1 = (a.bbox.y2 + a.bbox.y1)/2;
        center_x2 = (b.bbox.x2 + b.bbox.x1)/2;
        center_y2 = (b.bbox.y2 + b.bbox.y1)/2;
        distance_1 = (center_x1 - center_x)*(center_x1 - center_x) + (center_y1 - center_y)*(center_y1-center_y);
        distance_2 = (center_x2 - center_x)*(center_x2 - center_x) + (center_y2 - center_y)*(center_y1-center_y);
        return distance_1 < distance_2;
    default:
        return a.bbox.score > b.bbox.score;
    }
}

std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
                float thresh, nmsType nmstype){
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), cmpFunc(SCORE));

  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<bool> mask_merged(num_bbox, false);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx])
      select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = true;

    FaceRect select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.x2-select_bbox.x1+1) * (select_bbox.y2-select_bbox.y1+1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i])
        continue;

      FaceRect& bbox_i = bboxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0)
        continue;

      float area2 = static_cast<float>((bbox_i.x2-bbox_i.x1+1) * (bbox_i.y2-bbox_i.y1+1));
      float area_intersect = w * h;

      switch (nmstype) {
        case UNION:
          if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
            mask_merged[i] = true;
          break;
        case MINIMUM:
          if (static_cast<float>(area_intersect) / std::min(area1 , area2) > thresh)
            mask_merged[i] = true;
          break;
        default:
          break;
        }
    }
  }
  return bboxes_nms;
}

cv::Mat PCA(const cv::Mat& data,
        const cv::Mat& mean, const cv::Mat& evecs, int pcadim) {
    int w = data.cols;
    if (pcadim <= 0) {
        std::cout << "Wrong pcadim: " << pcadim << std::endl;
        std::cout << "Range is [1," << w << "]" << std::endl;
        std::cout << "Return self instead" << std::endl;
        return data;
    } else if (pcadim > w) {
        std::cout << "Wrong pcadim: " << pcadim << std::endl;
        std::cout << "Range is [1," << w << "]" << std::endl;
        std::cout << "Using " << w << " instead" << std::endl;
        pcadim = w;
    }
    cv::Rect roi(0, 0, w, pcadim);
    cv::Mat r;
    cv::PCAProject(data, mean, evecs(roi), r);
    return r;
}

float OneOne(const cv::Mat& feam1, const cv::Mat& feam2, bool l2norm) {
    cv::Mat feam1_norm, feam2_norm;
    if (l2norm) {
        float norm1 = cv::norm(feam1);
        float norm2 = cv::norm(feam2);
        feam1_norm = feam1 / norm1;
        feam2_norm = feam2 / norm2;
    } else {
        feam1_norm = feam1;
        feam2_norm = feam2;
    }
    float dis;
    dis = cv::norm(feam1_norm, feam2_norm);
    return dis;
}
float OneOne(float* fea1, float* fea2, int length, bool l2norm) {
    cv::Mat feam1(1, length, CV_32FC1, fea1);
    cv::Mat feam2(1, length, CV_32FC1, fea2);
    return OneOne(feam1, feam2, l2norm);
}

template <typename T>
cv::Mat Pointer2Mat(T* ptr, int height, int width, int type) {
    cv::Mat img(height, width, type, ptr);
    return img;
}
template cv::Mat Pointer2Mat(unsigned char* ptr, int h, int w, int t=CV_8UC3);
template cv::Mat Pointer2Mat(float* ptr, int h, int w, int t=CV_32FC3);

void drawFaceInfo(cv::Mat& img, const FaceInfo& faceInfo, bool drawPTs) {
    float x1 = faceInfo.bbox.x1;
    float y1 = faceInfo.bbox.y1;
    float x2 = faceInfo.bbox.x2;
    float y2 = faceInfo.bbox.y2;
    cv::rectangle(img, cv::Rect(x1, y1, x2-x1, y2-x1), cv::Scalar(0, 0, 255), 2);
    if (drawPTs) {
        for (int i = 0; i < 5; ++i) {
            cv::circle(img, cv::Point(faceInfo.facePts.x[i], faceInfo.facePts.y[i]),
                    2, cv::Scalar(0, 255, 0), -1);
        }
    }
}

void drawFaceInfo(cv::Mat& img, const std::vector<FaceInfo>& faceInfos, bool drawPTs) {
    for (size_t i = 0; i < faceInfos.size(); ++i) {
        drawFaceInfo(img, faceInfos[i], drawPTs);
    }
}

int txt2Mat(const std::string& txtFilePath, cv::Mat& mat,
        int rows, int cols, int c) {
    int retVal = 0;

    std::ifstream inFile(txtFilePath.c_str());
    if (!inFile.is_open()) {
        std::cout << "Open Failed: " << txtFilePath << std::endl;
        retVal = -1;
        return retVal;
    }
    std::istream_iterator<float> begin(inFile);
    std::istream_iterator<float> end;
    std::vector<float> inData(begin, end);
    cv::Mat tmpMat(inData);

    int dataLength = inData.size();
    if (c == 0) {
        c = 1;
    }
    if (rows != 0 && cols == 0) {
        cols = dataLength / rows / c;
    } else if (rows == 0 && cols != 0) {
        rows = dataLength / cols / c;
    } else if (rows == 0 && cols == 0) {
        rows = dataLength / c;
        cols = 1;
    }
    if (dataLength != (rows * cols * c)) {
        std::cout << "Imcompatible size: " << dataLength
            << " != " << rows << "*" << cols << "*" << c << std::endl;
        std::cout << "Using " << dataLength << "*1*1"
            << " instead" << std::endl;
        rows = dataLength;
        c = 1;
        retVal = 1;
    }

    mat = tmpMat.reshape(c, rows).clone();
    return retVal;
}

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    double r;
    r = 1000.0 * t.tv_sec + t.tv_usec / 1000.0;
    return r;
}
