/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: nyalg.cpp
 *     Created: 2017-06-17 16:18
 * Description:
 * ===================================================
 */
#include "detector.hpp"
#include "classifier.hpp"
#include "quality.hpp"
#include "property.hpp"
#include "ssd_landmark.h"
#include "metric.hpp"
#include "nypack.h"
#include "load.h"

typedef struct {
    const char *model;
    const char *featureModel;
    const char *featureTrained;
    const char *qualityModel;
    const char *qualityTrained;
    const char *propertyModel;
    const char *propertyTrained;
} NYConfig;

static NYConfig gcfg;

static bool GetNYUnpacker(const char *file, NYUnpacker *unpacker)
{
    std::string key(KKEY);
    if (key.empty()) {
        if (!NYGetKey(key)) {
            return false;
        }
    }

    const char *p[3] = {key.c_str(), KIV, KSLAT};
    for (int i=0; i<3; i++) {
        if (p[i] == NULL || strlen(p[i]) == 0) {
            return false;
        }
    }

    if (!unpacker->load(p, file)) {
        printf("load model:%s failed\n", file);
        return false;
    }

    return true;
}

class Config {
public:
    Config(): param_(NULL) {}

    bool Init(const char *file, const char *model) {
        if (!GetNYUnpacker(file, &unpacker_)) {
            return false;
        }

        if (!LoadConfig(unpacker_, &config_)) {
            return false;
        }

        param_ = FindParam(&config_, NULL, model);
        return !!param_;
    }

    cv::Size size() {
        return cv::Size(param_->size().width(), param_->size().height());
    }

    std::vector<cv::Point2f> pts() {
        std::vector<cv::Point2f> pts;
        for (int i=0; i<param_->pts_size(); i++) {
            pts.push_back(cv::Point2f(param_->pts(i).x(), param_->pts(i).y()));
        }
        return pts;
    }

    double bias() {
        return param_->bias();
    }

    double scale() {
        return param_->scale();
    }

    NYUnpacker* unpacker() {
        return &unpacker_;
    }

private:
    NYUnpacker unpacker_;
    model::config config_;
    const model::param *param_;
};

static bool use_ssd() {
    char *str = std::getenv("ALG_USE_SSD");
    return (str && std::string(str) == "1");
}

class IDetector {
public:
    virtual ~IDetector() {}
    virtual void init(NYUnpacker &unp, int device) = 0;
    virtual void detect(std::vector<FaceInfo> &result, const cv::Mat &mat, int miniSize, const float threshold[3]) = 0;
};

class NYSSDDetector: public IDetector {
public:
    NYSSDDetector():hSSDLM(NULL){}
    virtual ~NYSSDDetector() {if (hSSDLM) SSDLMDestroy(hSSDLM);}

    virtual void init(NYUnpacker &unp, int device) {
        m_ssd.init(unp, "VGG_face_SSD_300x300.prototxt", "VGG_face_SSD_300x300_iter_80000.caffemodel", device);
        m_ssd.prealloc();
        hSSDLM = SSDLMCreate(unp, "lbf_ssd300x300_celeba_5000.model");
    }

    virtual void detect(std::vector<FaceInfo> &result, const cv::Mat &mat, int miniSize, const float threshold[3]) {
        result = m_ssd.detect(mat, 0.1);
        for (auto &r : result) {
            SSDLMGetLandmark(hSSDLM, mat, r);
        }
    }

private:
    ssdDetector m_ssd;
    HSSDLM hSSDLM;
};

class NYMtcnnDetector: public IDetector {
public:
    virtual ~NYMtcnnDetector(){}
    virtual void init(NYUnpacker &unp, int device) {
        m_mtcnn.init(unp, device);
        m_mtcnn.prealloc();
    }

    virtual void detect(std::vector<FaceInfo> &result, const cv::Mat &mat, int miniSize, const float threshold[3]) {
        result = m_mtcnn.detect(mat, miniSize, (float*)threshold);
    }

private:
    mtcnnDetector m_mtcnn;
};

class NYDetector {
public:
    bool Init(const NYConfig *cfg, int device) {
        NYUnpacker unp;
        if (!GetNYUnpacker(cfg->model, &unp)) {
            printf("load model:%s failed\n", cfg->model);
            return false;
        }

        if (use_ssd()) m_detector.reset(new NYSSDDetector);
        else m_detector.reset(new NYMtcnnDetector);
        m_detector->init(unp, device);

        m_quality.init(unp, cfg->qualityModel, cfg->qualityTrained, device);
        m_quality.prealloc();

        m_property.init(unp, cfg->propertyModel, cfg->propertyTrained, device);
        m_property.prealloc();

        return true;
    }

    void Detect(const cv::Mat &mat, int miniSize, const float threshold[3], bool getProperty, std::vector<FaceInfo> &result) {
        m_detector->detect(result, mat, miniSize, threshold);

        for (auto &f : result) {
            if (f.bbox.x1 < 0) f.bbox.x1 = 0;
            if (f.bbox.y1 < 0) f.bbox.y1 = 0;
            if (f.bbox.x2 >= mat.size().width) f.bbox.x2 = mat.size().width - 1;
            if (f.bbox.y2 >= mat.size().height) f.bbox.y2 = mat.size().height - 1;
            cv::Mat subMat(mat, cv::Rect(cv::Point(f.bbox.x1, f.bbox.y1),
                        cv::Point(f.bbox.x2, f.bbox.y2)));
            f.quality = m_quality.detect(subMat);
            if (getProperty) {
                std::vector<float> property = m_property.detect(subMat);
                f.age = property[0];
                f.yaw = property[1];
                f.pitch = property[2];
                f.sex = property[3];
            }
        }
    }

private:
    std::shared_ptr<IDetector> m_detector;
    Quality m_quality;
    Property m_property;
};

// -------------------- API begin --------------------
#define API extern "C"

API int NYInit(const NYConfig *cfg)
{
    ::google::InitGoogleLogging("nyalg");
    gcfg = *cfg;
    return 0;
}

API int NYExit()
{
    return 0;
}

API void* NYDetectNew(int device)
{
    const NYConfig *cfg = &gcfg;
    NYDetector *ptr = new NYDetector();
    if (!ptr->Init(cfg, device)) {
        delete ptr;
        return NULL;
    }

    return ptr;
}

API void NYDetectDel(void*h)
{
    delete (NYDetector*)h;
}

API void NYDetect(void*h, const cv::Mat &mat, int minSize, const float threshold[3], bool getProperty, std::vector<FaceInfo> &result)
{
    ((NYDetector*)h)->Detect(mat, minSize, threshold, getProperty, result);
}

API void* NYExtractNew(int device)
{
    const NYConfig *cfg = &gcfg;
    Config config;
    if (!config.Init(cfg->model, cfg->featureTrained)) {
        printf("load config:%s failed\n", cfg->featureTrained);
        return NULL;
    }
    NYUnpacker &unp = *config.unpacker();

    Classifier *ptr = new Classifier;
    ptr->init(unp, cfg->featureModel, cfg->featureTrained, config.pts(), config.size(), config.bias(), config.scale(), device & 0xff);
    ptr->prealloc((device >> 8) & 0xff);
    return ptr;
}

API void NYExtractDel(void*h)
{
    delete (Classifier*)h;
}

API void NYExtract(void*h, const std::vector<cv::Mat> &mats, const std::vector<FaceInfo>& faceInfos, std::vector<std::vector<float> > &features)
{
    ((Classifier*)h)->extract(mats, faceInfos, features);
}

API int NYExtractDim(void*h)
{
    return ((Classifier*)h)->channels();
}
