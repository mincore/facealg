/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: detector.hpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

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

class mtcnnDetector {
 public:
  bool init(NYUnpacker &unp, int device = 0);
  void prealloc(int times=1);

  vector<FaceInfo> detect(const cv::Mat& img, int minSize,
          float threshold[], float factor=0.709, bool fastResize=false);
  vector<FaceInfo> detect(const cv::Mat& img, int minSize=40);
  template <typename T>
  vector<FaceInfo> detect(T* ptr,
          int height, int width, int type,
          int minSize, float threshold[], float factor=0.709,
          bool fastResize=false);
  template <typename T>
  vector<FaceInfo> detect(T* ptr,
          int height, int width, int type=CV_8UC3,
          int minSize=20);
 private:
  bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
  cv::Mat Preprocess(const cv::Mat& img);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels,Blob<float>* input_layer,
          const int height,const int width);
  void SetMean();
  void GenerateBoundingBox( Blob<float>* confidence,Blob<float>* reg,
          float scale,float thresh,int image_height,int image_width);
  void ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_float,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName);
  void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_float,
        boost::shared_ptr<Net<float> >& net, double thresh, char netName);
  void Bbox2Square(std::vector<FaceInfo>& bboxes);
  void Padding(int img_w, int img_h);
  std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);

 private:
  bool is_inited_;

  boost::shared_ptr<Net<float> > PNet_;
  boost::shared_ptr<Net<float> > RNet_;
  boost::shared_ptr<Net<float> > ONet_;

  // x1,y1,x2,t2 and score
  std::vector<FaceInfo> condidate_rects_;
  std::vector<FaceInfo> total_boxes_;
  std::vector<FaceInfo> regressed_rects_;
  std::vector<FaceInfo> regressed_pading_;

  int curr_feature_map_w_;
  int curr_feature_map_h_;
  int num_channels_;
};

class ssdDetector {
 public:
  bool init(NYUnpacker &unp, const string& modelFilePath,
          const string& trainedFilePath, int device=0);
  void prealloc(int times=1);

  std::vector<FaceInfo> detect(const cv::Mat& img, float threshold=0.1);
 private:
  void SetMean(const float mean_value[]);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  cv::Mat Preprocess(const cv::Mat& img);

 private:
  bool is_inited_;
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

#endif  // _DETECTOR_HPP_
