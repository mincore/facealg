/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: metric.cpp
 *     Created: 2017-11-07 16:45
 * Description:
 * ===================================================
 */
// c++
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>
//
#include <metric.hpp>

#include "nypack.h"

static void text_to_feature(const void *buf, size_t size, std::vector<float> &feat)
{
    const char *p = (const char *)buf;
    const char *s = p;
    const char *end = p + size;

    while (p < end) {
        while (p < end && isspace(*p)) p++;
        s = p;

        while (p < end && !isspace(*p)) p++;
        if (p > s) {
            feat.push_back(std::stof(std::string(s, p)));
        }
    }
}

static bool load_mat(NYUnpacker &unp, const std::string &name, int chan, int dim, cv::Mat &mat)
{
    std::vector<char> data;
    if (!unp.read(name.c_str(), data)) {
        printf("unpack %s failed\n", name.c_str());
        return false;
    }

    std::vector<float> inData;
    text_to_feature(&data[0], data.size(), inData);

    cv::Mat tmpMat(inData);
    mat = tmpMat.reshape(chan, dim).clone();
    return true;
}

bool Metric::init(NYUnpacker &unp,
                 const std::string& pcameanFilePath,
                 const std::string& evalsFilePath,
                 const std::string& evecsFilePath) {
    //txt2Mat(pcameanFilePath, pcamean_, 1, 0, 1);
    if (!load_mat(unp, pcameanFilePath, 1, 1, pcamean_)) {
        return false;
    }

    dim_ = pcamean_.cols;

    //txt2Mat(evalsFilePath, evals_, 1, 0, 1);
    if (!load_mat(unp, evalsFilePath, 1, dim_, evals_)) {
        return false;
    }

    //txt2Mat(evecsFilePath, evecs_, 0, dim_, 1);
    if (!load_mat(unp, evecsFilePath, 1, dim_, evecs_)) {
        return false;
    }

    return true;
}

void Metric::process(float* fea, int dim, std::vector<float> &out,
        int pcadim, bool l2norm) {
    cv::Mat feam(1, dim, CV_32FC1, fea);
    cv::Mat feam_pca;
    cv::Mat feam_norm;

    if (dim == dim_ && pcadim > 0 && pcadim <= dim_) {
        feam_pca = PCA(feam, pcamean_, evecs_, pcadim);
    } else {
        feam_pca = feam;
    }

    if (l2norm) {
        float norm = cv::norm(feam_pca);
        feam_norm = feam_pca / norm;
    } else {
        feam_norm = feam_pca;
    }

    float *p = (float*)feam_norm.data;
    out.assign(p, p + feam_norm.total());
}

void Metric::process(float* fea, int dim, std::vector<float> &out,
        double pcaratio, bool l2norm) {
    int pcadim;
    if (pcaratio <= 0) {
        pcadim = 0;
    } else if (pcaratio > 1) {
        pcadim = dim_;
    } else {
        float t = 0;
        float sum = cv::sum(evals_)[0];
        const float* data = evals_.ptr<float>(0);
        for (pcadim = 1; pcadim <= dim_; ++pcadim) {
            t += data[pcadim - 1];
            if (t >= pcaratio * sum) {
                break;
            }
        }
    }
    process(fea, dim, out, pcadim, l2norm);
}
