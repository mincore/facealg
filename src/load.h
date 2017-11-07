/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: load.h
 *     Created: 2017-07-31 10:49
 * Description:
 * ===================================================
 */
#ifndef _LOAD_H
#define _LOAD_H

#include <caffe/caffe.hpp>
#include "nypack.h"
#include "model.pb.h"

using namespace caffe;

bool LoadNet(
        NYUnpacker &unpakcer,
        boost::shared_ptr<Net<float> > &net,
        const std::string &protoName,
        const std::string &trainedName);

bool LoadConfig(
        NYUnpacker &unpakcer,
        model::config *config);

const model::param* FindParam(
        const model::config *config,
        const char *alias,
        const char *model);

#endif
