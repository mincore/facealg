/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: lbf.hpp
 *     Created: 2017-11-07 16:48
 * Description:
 * ===================================================
 */
#ifndef LBF_HPP_
#define LBF_HPP_

#include <vector>
#include "rf.hpp"
#include "faceinfo.hpp"
#include "../../src/nypack.h"

namespace lbf {

class LbfCascador {
public:
    void init(NYUnpacker &unp, const char* modelDirPath);
    void prealloc(int times=1);
    bool predict(cv::Mat &img, FaceInfo& faceInfo);

private:
    void Init(int stages_n);
    void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
               std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
               cv::Mat &mean_shape, int start_from = 0);
    void Test(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<BBox> &bboxes);
    void GlobalRegressionTrain(std::vector<cv::Mat> &lbfs, std::vector<cv::Mat> &deltashapes, int stage);
    cv::Mat GlobalRegressionPredict(const cv::Mat &lbf, int stage);
    void DumpTrainModel(int stage);
    void ResumeTrainModel(int start_from = 0);

    void Write(FILE *fd);

public:
    int stages_n;
    int landmark_n;
    cv::Mat mean_shape;
    std::vector<RandomForest> random_forests;
    std::vector<cv::Mat> gl_regression_weights;
};

} // namespace lbf

#endif // LBF_HPP_
