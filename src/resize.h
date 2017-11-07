/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: resize.h
 *     Created: 2017-10-09 17:49
 * Description:
 * ===================================================
 */
#ifndef _RESIZE_H
#define _RESIZE_H

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void CudaResize(const cv::Mat &src, cv::Mat &dst, cv::Size size);

#endif
