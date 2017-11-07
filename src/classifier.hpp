/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: classifier.hpp
 *     Created: 2017-11-07 16:43
 * Description:
 * ===================================================
 */
#ifndef _CLASSIFIER_HPP_
#define _CLASSIFIER_HPP_

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

class Classifier {
  public:
    bool init(NYUnpacker &unp,
            const std::string& modelFilePath, const std::string& trainedFilePath,
            const std::vector<cv::Point2f>& pts, const cv::Size& dsize,
            double bias, double scale, int device=0);
    void prealloc(int num, int times=1);

    int channels();
    bool extract(const std::vector<cv::Mat>& imgs,
            const std::vector<FaceInfo>& faceInfos,
            std::vector<std::vector<float> >& feas,
            bool whetherAlign=true);
    bool extract(const cv::Mat& img,
            const FaceInfo& faceInfo,
            std::vector<float>& fea,
            bool whetherAlign=true);
    bool extract(const std::vector<cv::Mat>& imgs,
            std::vector<std::vector<float> >& feas);
    bool extract(const cv::Mat& img,
            std::vector<float>& fea);

  private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels, int n);

    cv::Mat Align(const cv::Mat& img, const FacePts& facePts);
    cv::Mat Crop(const cv::Mat& img, const cv::Size& input_size);

    cv::Mat Preprocess(const cv::Mat& img,
            const FaceInfo& faceInfo,
            bool whetherAlign=true);
    cv::Mat Preprocess(const cv::Mat& img);

  private:
    bool is_inited_;
    int dim_;
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    std::vector<cv::Point2f> dst_pts_;
    cv::Size aligned_size_;
    double bias_;
    double scale_;
};

#endif  // _CLASSIFIER_HPP_
