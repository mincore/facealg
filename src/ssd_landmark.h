/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: ssd_landmark.h
 *     Created: 2017-07-07 10:31
 * Description:
 * ===================================================
 */
#ifndef _SSD_LANDMARK_H
#define _SSD_LANDMARK_H

#include "nypack.h"

#define API extern "C"

typedef void* HSSDLM;

API HSSDLM SSDLMCreate(NYUnpacker &unp, const char *model);
API void SSDLMDestroy(HSSDLM);
API void SSDLMGetLandmark(HSSDLM, const cv::Mat &mat, FaceInfo &faceInfo);

#endif
