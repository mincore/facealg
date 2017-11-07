/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: ssd_landmark.cpp
 *     Created: 2017-07-07 17:42
 * Description:
 * ===================================================
 */
#include "lbf.hpp"

#define API extern "C"

typedef void* HSSDLM;

using namespace lbf;

API HSSDLM SSDLMCreate(NYUnpacker &unp, const char *model)
{
    LbfCascador *ptr = new LbfCascador;
    ptr->init(unp, model);
    ptr->prealloc();
    return (HSSDLM)ptr;
}

API void SSDLMDestroy(HSSDLM h)
{
    delete (LbfCascador*)h;
}

API void SSDLMGetLandmark(HSSDLM h, const cv::Mat &mat, FaceInfo &faceInfo)
{
    LbfCascador *ptr = (LbfCascador*)h;
    ptr->predict((cv::Mat&)mat, faceInfo);
}

