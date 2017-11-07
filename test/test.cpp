/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: test.cpp
 *     Created: 2017-07-04 11:25
 * Description:
 * ===================================================
 */
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <atomic>
#include <thread>
#include <unistd.h>
#include "nyalg.h"

#define MINISIZE 60
const float THRESHOLD[3] = { 0.8, 0.7, 0.7 };

static void Show(const char *file, cv::Mat &mat, std::vector<FaceInfo> &infos)
{
    char text[256];
    int i=0;

    for (auto &info : infos) {
        printf("[%02d] rect: %f %f %f %f, prof:%f, %f, %f, %f, %f\n",
                i,
                info.bbox.x1, info.bbox.y1,
                info.bbox.x2, info.bbox.y2,
                info.age, info.sex, info.yaw, info.pitch,
                info.quality);

        snprintf(text, sizeof(text), "%d", i);
        cv::Rect rc(cv::Point(info.bbox.x1, info.bbox.y1), cv::Point(info.bbox.x2, info.bbox.y2));
        cv::rectangle(mat, rc, cv::Scalar(0, 255, 0), 2);
        cv::putText(mat, text, cv::Point(info.bbox.x1, info.bbox.y1),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 200, 200), 2);
        i++;
    }

    cv::imwrite(file, mat);
}

static void ReadVideo(void*d, const char *videoFile)
{
    cv::VideoCapture cap(videoFile);
    if(!cap.isOpened()) {
        printf("open %s failed\n", videoFile);
        return;
    }

    int i = 0;
    cv::Mat mat;
    char file[256];

    while (cap.read(mat)) {
        std::vector<FaceInfo> infos;
        NYDetect(d, mat, MINISIZE, THRESHOLD, true, infos);

        snprintf(file, sizeof(file), "./%d.jpg", i);
        Show(file, mat, infos);
        i++;
    }
}

static void ReadImgs(void*d, void*e, int argc, char **argv)
{
    char file[256];

    //int n = 0;
    //char *str = std::getenv("N");
    //if (str)
    //    n = std::atoi(str);

    for (int i=0; i<argc; i++) {
        cv::Mat mat = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
        std::vector<FaceInfo> infos;
        NYDetect(d, mat, MINISIZE, THRESHOLD, true, infos);
        snprintf(file, sizeof(file), "./%d.jpg", i);
        Show(file, mat, infos);

        std::vector<cv::Mat> mats;
        std::vector<std::vector<float> > feats;
        std::vector<FaceInfo> newinfos;

        for (size_t i=0; i<infos.size(); i++) {
            mats.push_back(mat);
            newinfos.push_back(infos[i]);
        }

        auto t1 = now();
        NYExtract(e, mats, newinfos, feats);
        printf("extract %zd use %lldms\n", infos.size(), now()-t1);

        if (infos.size() >= 2) {
            printf("feature size: %zd, cmp: %f\n", feats[0].size(),
                    cv::norm(cv::Mat(feats[0]), cv::Mat(feats[1])));
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
        return -1;

    NYConfig cfg;

    cfg.model = "./model";
    cfg.featureModel = "sphereface_ave_norm.prototxt";
    cfg.featureTrained = "sphereface_sphereface_size112x96_30-52_66-52_48-92_28000.caffemodel";
    cfg.qualityModel = "qa_deploy.prototxt";
    cfg.qualityTrained = "qa_iter_50000.caffemodel";
    cfg.propertyModel = "prop_deploy.prototxt";
    cfg.propertyTrained = "prop_iter_50000.caffemodel";

    int device = 0;
    char *str = std::getenv("DEVICE");
    if (str)
        device = std::atoi(str);

    NYInit(&cfg);

    NYDetector detector(device);
    NYExtractor extractor(device);

    if (argc >= 3 && std::string(argv[1]) == "-v") {
        ReadVideo(detector, argv[2]);
    } else {
        ReadImgs(detector, extractor, argc-1, &argv[1]);
    }

    NYExit();

    return 0;
}
