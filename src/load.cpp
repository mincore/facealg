/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: load.cpp
 *     Created: 2017-07-31 11:43
 * Description:
 * ===================================================
 */
#include "load.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::Message;

static inline bool ReadTextProtoFromMem(const void *buf, size_t size, Message *proto) {
    return google::protobuf::TextFormat::ParseFromString(std::string((char*)buf, size), proto);
}

static inline bool ReadBinaryProtoFromMem(const void* buf, size_t size, Message* proto) {
  ArrayInputStream input(buf, size);
  CodedInputStream coded_input(&input);
  coded_input.SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
  return proto->ParseFromCodedStream(&coded_input);
}

bool LoadNet(
        NYUnpacker &unpakcer,
        boost::shared_ptr<Net<float> > &net,
        const std::string &protoName,
        const std::string &trainedName)
{
    NetParameter param;
    std::vector<char> data;

    if (!unpakcer.read(protoName.c_str(), data)) {
        printf("unpack proto:%s failed\n", protoName.c_str());
        return false;
    }

    if (!ReadTextProtoFromMem(&data[0], data.size(), &param)) {
        printf("parse failed\n");
        return false;
    }

    param.mutable_state()->set_phase(TEST);
    param.mutable_state()->set_level(0);
    UpgradeNetAsNeeded(protoName.c_str(), &param);

    net.reset(new Net<float>(param));

    data.clear();
    if (!unpakcer.read(trainedName.c_str(), data)) {
        printf("unpack model:%s failed\n", trainedName.c_str());
        return false;
    }

    if (!ReadBinaryProtoFromMem(&data[0], data.size(), &param)) {
        printf("parse failed\n");
        return false;
    }
    net->CopyTrainedLayersFrom(param);

    return true;
}

bool LoadConfig(
        NYUnpacker &unpakcer,
        model::config *config)
{
    std::vector<char> data;
    if (!unpakcer.read("model.prototxt", data)) {
        printf("unpack model.prototxt failed\n");
        return false;
    }

    if (!ReadTextProtoFromMem(&data[0], data.size(), config)) {
        printf("parse failed\n");
        return false;
    }

    return true;
}

const model::param* FindParam(
        const model::config *config,
        const char *alias,
        const char *model)
{
    if (alias) {
        for (int i=0; i<config->params_size(); i++) {
            if (config->params(i).alias() == alias)
                return &config->params(i);
        }
    }

    if (model) {
        for (int i=0; i<config->params_size(); i++) {
            if (config->params(i).model() == model)
                return &config->params(i);
        }
    }

    return NULL;
}
