/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: faceutil.hpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
#ifndef _FACEUTIL_HPP_
#define _FACEUTIL_HPP_

#include <faceinfo.hpp>

/*
enum cmpType{SCORE, AREA, CENTER};
class cmpFunc
{
  public:
    cmpFunc(cmpType c, int height, int width):cmptype(c),h(height),w(width) {}
    cmpFunc(cmpType c) {cmpFunc(c, 0, 0);}
    cmpFunc() {cmpFunc(SCORE);}
    bool operator () (const FaceInfo& a, const FaceInfo& b) {
      float area1, area2;
      float center_x, center_y;
      float center_x1, center_y1;
      float center_x2, center_y2;
      float distance_1, distance_2;
      switch(cmptype) {
        case SCORE:
          return a.bbox.score>b.bbox.score;
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
          return a.bbox.score>b.bbox.score;
      }
    }
  private:
    cmpType cmptype;
    float h, w;
};
*/

bool cmpFunc_SCORE(const FaceInfo& a, const FaceInfo& b) {
    return a.bbox.score > b.bbox.score;
}
bool cmpFunc_AREA(const FaceInfo& a, const FaceInfo& b) {
    float area1 = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
    float area2 = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
    return area1 > area2;
}

std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
                float thresh,char nmstype){
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), cmpFunc_SCORE);

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
        case 'u':
          if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
            mask_merged[i] = true;
          break;
        case 'm':
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
// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Minimum)
enum nmsType{UNION, MINIMUM};
std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
                float thresh, nmsType nmstype){
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), cmpFunc_SCORE);

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

float OneOne(const std::vector<float>& fea1, const std::vector<float>& fea2,
        bool l2norm=true) {
    float dis = 0;
    if (fea1.empty()) {
        dis += -1;
    }
    if (fea2.empty()) {
        dis += -2;
    }
    if (dis) {
        return dis;
    }
    cv::Mat feam1(fea1), feam2(fea2);
    if (l2norm) {
        float norm1 = cv::norm(feam1);
        float norm2 = cv::norm(feam2);
        feam1 /= norm1;
        feam2 /= norm2;
    }
    dis = cv::norm(feam1, feam2);
    return dis;
}

template <typename T>
cv::Mat Pointer2Mat(T* ptr, int height, int width, int type=CV_8UC3) {
    cv::Mat img(height, width, type, ptr);
    return img;
}

void drawFaceInfo(cv::Mat& img, const FaceInfo& faceInfo, bool drawPTs=true) {
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

void drawFaceInfo(cv::Mat& img, const std::vector<FaceInfo>& faceInfos, bool drawPTs=true) {
    for (int i = 0; i < faceInfos.size(); ++i) {
        drawFaceInfo(img, faceInfos[i], drawPTs);
    }
}

#endif  // _FACEUTIL_HPP_
