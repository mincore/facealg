/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: classifier.cpp
 *     Created: 2017-11-07 16:43
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
//
#include <classifier.hpp>

#include "load.h"

bool Classifier::init(NYUnpacker &unp,
                 const std::string& modelFilePath,
                 const std::string& trainedFilePath,
                 const std::vector<cv::Point2f>& pts,
                 const cv::Size& dsize,
                 double bias,
                 double scale,
                 int device) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::SetDevice(device);
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    if (!LoadNet(unp, net_, modelFilePath, trainedFilePath)) {
        printf("load net failed\n");
        return false;
    }

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    dst_pts_.assign(pts.begin(), pts.end());
    CHECK(dst_pts_.size() == 3 || dst_pts_.size() == 5)
        << "Align template should have 3 or 5 points.";
    aligned_size_ = dsize;
    bias_ = bias;
    scale_ = scale;

    input_layer->Reshape(1, num_channels_,
            input_geometry_.height, input_geometry_.width);
    // Forward dimension change to all layers.
    net_->Reshape();

    Blob<float>* output_layer = net_->output_blobs()[0];
    dim_ = output_layer->channels();

    is_inited_ = true;
    return is_inited_;
}

void Classifier::prealloc(int num, int times) {
    if (!is_inited_) {
        std::cout << "prealloc: Classifier Not Inited." << std::endl;
        return;
    }
    if (num == 0) num = 1;
    std::vector<cv::Mat> imgs;
    std::vector<FaceInfo> faceInfos;
    for (int i = 0; i < num; ++i) {
        cv::Mat img(input_geometry_, CV_32FC3);
        cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
        imgs.push_back(img);
        FaceInfo faceInfo;
        faceInfos.push_back(faceInfo);
    }
    std::vector<std::vector<float> > feas;
    for (int i = 0; i < times; ++i) {
        extract(imgs, faceInfos, feas, false);
    }
}

int Classifier::channels() {
    if (!is_inited_) {
        std::cout << "channels: Classifier Not Inited." << std::endl;
        return 0;
    }
    return dim_;
}

bool Classifier::extract(const std::vector<cv::Mat>& imgs,
        const std::vector<FaceInfo>& faceInfos,
        std::vector<std::vector<float> >& feas,
        bool whetherAlign) {
    if (!is_inited_) {
        std::cout << "extract: Classifier Not Inited." << std::endl;
        return false;
    }
    if (faceInfos.size() != imgs.size()) {
        std::cout << "Number of faceInfos should be equal to number of imgs." << std::endl;
        return false;
    }

    Blob<float>* input_layer = net_->input_blobs()[0];
    if (input_layer->num() != imgs.size()) {
        input_layer->Reshape(imgs.size(), num_channels_,
                input_geometry_.height, input_geometry_.width);
        // Forward dimension change to all layers.
        net_->Reshape();
    }
#ifndef NDEBUG
    double t1 = getTime();
#endif
    std::vector<cv::Mat> samples;
    cv::Mat sample;
    for (int i = 0; i < imgs.size(); ++i) {
        sample = Preprocess(imgs[i], faceInfos[i], whetherAlign);
        samples.push_back(sample);
    }
#ifndef NDEBUG
    double t2 = getTime();
    std::cout << "Time of Classifier PREPROCESS: " << t2-t1 << std::endl;
#endif
    for (int i = 0; i < samples.size(); ++i) {
        /* This operation will write the separate BGR planes directly to the
        * input layer of the network because it is wrapped by the cv::Mat
        * objects in input_channels. */
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels, i);

        cv::Mat sample = samples[i];
        cv::split(sample, input_channels);

        CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
                == net_->input_blobs()[0]->cpu_data()
                    + net_->input_blobs()[0]->offset(i))
            << "Input channels are not wrapping the input layer of the network.";
    }
    net_->Forward();
#ifndef NDEBUG
    double t3 = getTime();
    std::cout << "Time of Classifier FORWARD: " << t3-t2 << std::endl;
#endif

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    for (int i = 0; i < samples.size(); ++i) {
        std::vector<float> fea(begin + output_layer->offset(i),
                begin + output_layer->offset(i+1));
        feas.push_back(fea);
    }
#ifndef NDEBUG
    double t4 = getTime();
    std::cout << "Time of Classifier COPY: " << t4-t3 << std::endl;
#endif
    return true;
}
bool Classifier::extract(const cv::Mat& img,
        const FaceInfo& faceInfo,
        std::vector<float>& fea,
        bool whetherAlign) {
    std::vector<cv::Mat> imgs;
    std::vector<FaceInfo> faceInfos;
    imgs.push_back(img);
    faceInfos.push_back(faceInfo);
    std::vector<std::vector<float> > feas;
    bool status = extract(imgs, faceInfos, feas, whetherAlign);
    if (status) {
        fea = feas[0];
    }
    return status;
}

bool Classifier::extract(const std::vector<cv::Mat>& imgs,
        std::vector<std::vector<float> >& feas) {
    std::vector<FaceInfo> faceInfos;
    for (int i = 0; i < imgs.size(); ++i) {
        FaceInfo faceInfo;
        faceInfos.push_back(faceInfo);
    }
    return extract(imgs, faceInfos, feas, false);
}
bool Classifier::extract(const cv::Mat& img,
        std::vector<float>& fea) {
    FaceInfo faceInfo;
    return extract(img, faceInfo, fea, false);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int n) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data() + input_layer->offset(n);
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

cv::Mat Classifier::Align(const cv::Mat& img,
                          const FacePts& facePts) {
    //left eye
    float left_eye_x = facePts.x[0];
    float left_eye_y = facePts.y[0];
    //right eye
    float right_eye_x = facePts.x[1];
    float right_eye_y = facePts.y[1];
    //nose
    float nose_x = facePts.x[2];
    float nose_y = facePts.y[2];
    //mouth left
    float mouth_left_x = facePts.x[3];
    float mouth_left_y = facePts.y[3];
    //mouth right
    float mouth_right_x = facePts.x[4];
    float mouth_right_y = facePts.y[4];
    //mouth center
    float mouth_center_x = (mouth_left_x + mouth_right_x) / 2;
    float mouth_center_y = (mouth_left_y + mouth_right_y) / 2;

    cv::Mat affineMat;
    std::vector<cv::Point2f> src_pts;
    if (dst_pts_.size() == 3) {
        src_pts.push_back(cv::Point2f(left_eye_x, left_eye_y));
        src_pts.push_back(cv::Point2f(right_eye_x, right_eye_y));
        src_pts.push_back(cv::Point2f(mouth_center_x, mouth_center_y));
        affineMat = cv::getAffineTransform(src_pts, dst_pts_);
    } else if (dst_pts_.size() == 5) {
        /*
        src_pts.push_back(cv::Point2f(left_eye_x, left_eye_y));
        src_pts.push_back(cv::Point2f(right_eye_x, right_eye_y));
        src_pts.push_back(cv::Point2f(nose_x, nose_y));
        src_pts.push_back(cv::Point2f(mouth_left_x, mouth_left_y));
        src_pts.push_back(cv::Point2f(mouth_right_x, mouth_right_y));
        affineMat = cv::estimateRigidTransform(src_pts, dst_pts_, false);
        */
        //cv::estimateRigidTransform may return an empty Mat
        std::vector<cv::Point2f> src_pts_3;
        src_pts_3.push_back(cv::Point2f(left_eye_x, left_eye_y));
        src_pts_3.push_back(cv::Point2f(right_eye_x, right_eye_y));
        src_pts_3.push_back(cv::Point2f(mouth_center_x, mouth_center_y));
        std::vector<cv::Point2f> dst_pts_3;
        dst_pts_3.push_back(dst_pts_[0]);
        dst_pts_3.push_back(dst_pts_[1]);
        float tx = (dst_pts_[3].x + dst_pts_[4].x) / 2;
        float ty = (dst_pts_[3].y + dst_pts_[4].y) / 2;
        cv::Point2f t(tx, ty);
        dst_pts_3.push_back(t);
        affineMat = cv::getAffineTransform(src_pts_3, dst_pts_3);
    }

    cv::Mat alignedImg;
    cv::warpAffine(img, alignedImg, affineMat, aligned_size_,
            cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return alignedImg;
}

cv::Mat Classifier::Crop(const cv::Mat& img, const cv::Size& input_size) {
    cv::Mat cropped;
    int height = img.rows;
    int width = img.cols;
    CHECK(height >= input_size.height)
        << "Crop Error: the height of image should be larger than input_size.";
    CHECK(width >= input_size.width)
        << "Crop Error: the weight of image should be larger than input_size.";
    if (height == input_size.height && width == input_size.width) {
        cropped = img;
    } else {
        float y = ((height - input_size.height) / 2);
        float x = ((width - input_size.width) / 2);
        cv::Rect roi(cv::Point2f(x, y), input_size);
        cropped = img(roi);
    }
    return cropped;
}

cv::Mat Classifier::Preprocess(const cv::Mat& img,
                               const FaceInfo& faceInfo,
                               bool whetherAlign) {
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

    cv::Mat sample_aligned;
    if (whetherAlign) {
        sample_aligned = Align(sample, faceInfo.facePts);
    } else {
        sample_aligned = sample;
    }
    cv::Mat sample_cropped;
    sample_cropped = Crop(sample_aligned, input_geometry_);
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_cropped.convertTo(sample_float, CV_32FC3, scale_, -scale_*bias_);
    else
        sample_cropped.convertTo(sample_float, CV_32FC1, scale_, -scale_*bias_);

    return sample_float;
}

cv::Mat Classifier::Preprocess(const cv::Mat& img) {
    FaceInfo faceInfo;
    return Preprocess(img, faceInfo, false);
}
