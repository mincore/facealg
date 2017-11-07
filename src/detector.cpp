/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: detector.cpp
 *     Created: 2017-11-07 16:44
 * Description:
 * ===================================================
 */
// caffe
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
// c++
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>

#include "load.h"

using namespace cv;

bool mtcnnDetector::init(NYUnpacker &unp, int device){
  Caffe::SetDevice(device);
  Caffe::set_mode(Caffe::GPU);

  //PNet_.reset(new Net<float>((modelDirPath+"/det1_input.prototxt"), TEST));
  //PNet_->CopyTrainedLayersFrom(modelDirPath+"/det1.caffemodel");
  if (!LoadNet(unp, PNet_, "det1_input.prototxt", "det1.caffemodel")) {
      printf("load net failed\n");
      return false;
  }

  CHECK_EQ(PNet_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(PNet_->num_outputs(),2) << "Network should have exactly two output,"
                                      " one is bbox and another is confidence.";

  //RNet_.reset(new Net<float>((modelDirPath+"/det2_input.prototxt"), TEST));
  //RNet_->CopyTrainedLayersFrom(modelDirPath+"/det2.caffemodel");
  if (!LoadNet(unp, RNet_, "det2_input.prototxt", "det2.caffemodel")) {
      printf("load net failed\n");
      return false;
  }

  //ONet_.reset(new Net<float>((modelDirPath+"/det3_input.prototxt"), TEST));
  //ONet_->CopyTrainedLayersFrom(modelDirPath+"/det3.caffemodel");
  if (!LoadNet(unp, ONet_, "det3_input.prototxt", "det3.caffemodel")) {
      printf("load net failed\n");
      return false;
  }

  Blob<float>* input_layer;
  input_layer = PNet_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  is_inited_ = true;
  return is_inited_;
}

void mtcnnDetector::prealloc(int times) {
  if (!is_inited_) {
    std::cout << "mtcnnDetector Not Inited." << std::endl;
    return;
  }
  cv::Mat t;
  t = cv::Mat::zeros(100,100,CV_32FC3);
  detect(t);
}

std::vector<FaceInfo> mtcnnDetector::detect(const cv::Mat& img,
        int minSize, float threshold[], float factor, bool fastResize){
  if (!is_inited_) {
    std::cout << "mtcnnDetector Not Inited." << std::endl;
    std::vector<FaceInfo> t;
    return t;
  }
#ifndef NDEBUG
  std::cout << "size: " << img.rows << "X" << img.cols << std::endl;
  double t1 = getTime();
#endif

  cv::Mat sample_float,resized;
  sample_float = Preprocess(img);
  sample_float = sample_float.t();

#ifndef NDEBUG
  double t2 = getTime();
  std::cout << "Time of mtcnnDetector PREPROCESS: " << t2-t1 << std::endl;
#endif

  int height = sample_float.rows;
  int width  = sample_float.cols;

  int minWH = std::min(height,width);
  double m = 12./minSize;
  minWH *= m;
  std::vector<double> scales;
  double cur_scale = 1.0;
  while (minWH >= 12)
  {
    scales.push_back(m * cur_scale);
    cur_scale *= factor;
    minWH *= factor;
  }

  Blob<float>* input_layer = PNet_->input_blobs()[0];

  std::vector<Mat> resizedMats;

#if 0
  extern void CoScale(const cv::Mat &mat, const std::vector<double> &scales,
        std::vector<cv::Mat> &resizedMats);
  CoScale(sample_float, scales, resizedMats);
#else
  resizedMats.resize(scales.size());
  for(size_t i=0;i<scales.size();i++)
  {
    double scale = scales[i];
    int hs = std::ceil(height*scale);
    int ws = std::ceil(width*scale);
    Mat &resized = resizedMats[i];

    fastResize = 0;
    if (fastResize) {
      sample_float.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);
      cv::resize(resized,resized,cv::Size(ws,hs));
    } else {
      cv::resize(sample_float,resized,cv::Size(ws,hs));
      resized.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);
    }
  }
#endif

  for(size_t i=0;i<scales.size();i++)
  {
    double scale = scales[i];
    int hs = std::ceil(height*scale);
    int ws = std::ceil(width*scale);
    Mat &resized = resizedMats[i];

    // input data
    input_layer->Reshape(1, 3, hs, ws);
    PNet_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels,PNet_->input_blobs()[0],hs,ws);
    cv::split(resized,input_channels);

    // check data transform right
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
            == PNet_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    PNet_->Forward();

    // return result
    Blob<float>* reg = PNet_->output_blobs()[0];
    Blob<float>* confidence = PNet_->output_blobs()[1];
    GenerateBoundingBox(confidence, reg, scale, threshold[0], hs, ws);
    std::vector<FaceInfo> bboxes_nms = NMS(condidate_rects_,0.5,UNION);
    total_boxes_.insert(total_boxes_.end(),bboxes_nms.begin(),bboxes_nms.end());
  }
  int numBox = total_boxes_.size();

#ifndef NDEBUG
  double t3 = getTime();
  double t4, t5;
  std::cout << "Time of mtcnnDetector STAGE1: " << t3-t2 << " " << numBox << std::endl;
#endif
  std::vector<FaceInfo> faceInfos;
  if(numBox != 0){
    total_boxes_ = NMS(total_boxes_,0.7,UNION);
    regressed_rects_ = BoxRegress(total_boxes_,1);
    total_boxes_.clear();

    Bbox2Square(regressed_rects_);
    Padding(height, width);

    /// Second stage
    ClassifyFace_MulImage(regressed_rects_,sample_float,RNet_,threshold[1],'r');
    condidate_rects_ = NMS(condidate_rects_,0.7,UNION);
    regressed_rects_ = BoxRegress(condidate_rects_,2);

    Bbox2Square(regressed_rects_);
    Padding(height, width);
    numBox = regressed_rects_.size();

#ifndef NDEBUG
    t4 = getTime();
    std::cout << "Time of mtcnnDetector STAGE2: " << t4-t3 << " " << numBox << std::endl;
#endif
    /// three stage
    if(numBox != 0){
      ClassifyFace_MulImage(regressed_rects_,sample_float,ONet_,threshold[2],'o');
      regressed_rects_ = BoxRegress(condidate_rects_,3);
      faceInfos = NMS(regressed_rects_,0.7,MINIMUM);
    }
#ifndef NDEBUG
    t5 = getTime();
    std::cout << "Time of mtcnnDetector STAGE3: " << t5-t4 << " "<< faceInfos.size() <<std::endl;
#endif
  }
  regressed_pading_.clear();
  regressed_rects_.clear();
  condidate_rects_.clear();

#ifndef NDEBUG
  t5 = getTime();
#endif
  for (size_t i = 0; i < faceInfos.size(); ++i) {
    float t;
    t = faceInfos[i].bbox.x1;
    faceInfos[i].bbox.x1 = faceInfos[i].bbox.y1;
    faceInfos[i].bbox.y1 = t;
    t = faceInfos[i].bbox.x2;
    faceInfos[i].bbox.x2 = faceInfos[i].bbox.y2;
    faceInfos[i].bbox.y2 = t;
    for (int j = 0; j < 5; ++j) {
      t = faceInfos[i].facePts.x[j];
      faceInfos[i].facePts.x[j] = faceInfos[i].facePts.y[j];
      faceInfos[i].facePts.y[j] = t;
    }
  }
  sort(faceInfos.begin(), faceInfos.end(), cmpFunc(AREA));
#ifndef NDEBUG
  double t6 = getTime();
  std::cout << "Time of mtcnnDetector SWAP: " << t6-t5 << std::endl;
#endif
  return faceInfos;
}
std::vector<FaceInfo> mtcnnDetector::detect(const cv::Mat& img,
        int minSize){
  float threshold[3] = {0.6, 0.7, 0.7};
  return detect(img, minSize, threshold);
}

template <typename T>
std::vector<FaceInfo> mtcnnDetector::detect(T* ptr,
        int height, int width, int type,
        int minSize, float threshold[], float factor, bool fastResize) {
  cv::Mat img = Pointer2Mat(ptr, height, width, type);
  return detect(img, minSize, threshold, factor, fastResize);
}
template <typename T>
std::vector<FaceInfo> mtcnnDetector::detect(T* ptr,
        int height, int width, int type, int minSize) {
  cv::Mat img = Pointer2Mat(ptr, height, width, type);
  return detect(img, minSize);
}
template std::vector<FaceInfo> mtcnnDetector::detect<uchar>(uchar* ptr,
        int height, int width, int type,
        int minSize, float threshold[], float factor, bool fastResize);
template std::vector<FaceInfo> mtcnnDetector::detect<uchar>(uchar* ptr,
        int height, int width, int type,
        int minSize);

void mtcnnDetector::Bbox2Square(std::vector<FaceInfo>& bboxes){
  for(size_t i=0;i<bboxes.size();i++){
    float w = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
    float h = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
    float side = h>w ? h:w;
    bboxes[i].bbox.x1 += (w-side)*0.5;
    bboxes[i].bbox.y1 += (h-side)*0.5;

    bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
    bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
    bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
    bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

  }
}

std::vector<FaceInfo> mtcnnDetector::BoxRegress(std::vector<FaceInfo>& faceInfo,int stage){
  std::vector<FaceInfo> bboxes;
  for(size_t bboxId =0;bboxId<faceInfo.size();bboxId++){
      FaceRect faceRect;
      FaceInfo tempFaceInfo;
      float regh = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
      regh += (stage == 1)? 0:1;
      float regw = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
      regw += (stage == 1)? 0:1;
      faceRect.y1 = faceInfo[bboxId].bbox.y1 + regh * faceInfo[bboxId].regression[0];
      faceRect.x1 = faceInfo[bboxId].bbox.x1 + regw * faceInfo[bboxId].regression[1];
      faceRect.y2 = faceInfo[bboxId].bbox.y2 + regh * faceInfo[bboxId].regression[2];
      faceRect.x2 = faceInfo[bboxId].bbox.x2 + regw * faceInfo[bboxId].regression[3];
      faceRect.score = faceInfo[bboxId].bbox.score;

      tempFaceInfo.bbox = faceRect;
      tempFaceInfo.regression = faceInfo[bboxId].regression;
      if(stage == 3)
        tempFaceInfo.facePts = faceInfo[bboxId].facePts;
      bboxes.push_back(tempFaceInfo);
  }
  return bboxes;
}

// compute the padding coordinates (pad the bounding boxes to square)
void mtcnnDetector::Padding(int img_h,int img_w){
  for(size_t i=0;i<regressed_rects_.size();i++){
    FaceInfo tempFaceInfo;
    tempFaceInfo = regressed_rects_[i];
    tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
    tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
    tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_w) ? img_w : regressed_rects_[i].bbox.x2;
    tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_h) ? img_h : regressed_rects_[i].bbox.y2;
    regressed_pading_.push_back(tempFaceInfo);
  }
}

void mtcnnDetector::GenerateBoundingBox(Blob<float>* confidence,Blob<float>* reg,
      float scale,float thresh,int image_height,int image_width){
  int stride = 2;
  int cellSize = 12;

  //int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0/stride)+1;
  //int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0/stride)+1;
  int curr_feature_map_h_ = confidence->height();
  int curr_feature_map_w_ = confidence->width();

  int regOffset = curr_feature_map_w_*curr_feature_map_h_;
  int count = confidence->count()/2;
  const float* confidence_data = confidence->cpu_data();
  confidence_data += count;
  const float* reg_data = reg->cpu_data();

  condidate_rects_.clear();
  for(int i=0;i<count;i++){
    if(*(confidence_data+i)>=thresh){
      int y = i / curr_feature_map_w_;
      int x = i - curr_feature_map_w_ * y;

      float xTop = (int)((x*stride+1)/scale);
      float yTop = (int)((y*stride+1)/scale);
      float xBot = (int)((x*stride+cellSize-1+1)/scale);
      float yBot = (int)((y*stride+cellSize-1+1)/scale);
      FaceRect faceRect;
      faceRect.x1 = xTop;
      faceRect.y1 = yTop;
      faceRect.x2 = xBot;
      faceRect.y2 = yBot;
      faceRect.score  = *(confidence_data+i);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[i+0*regOffset],reg_data[i+1*regOffset],reg_data[i+2*regOffset],reg_data[i+3*regOffset]);
      condidate_rects_.push_back(faceInfo);
    }
  }
}

cv::Mat mtcnnDetector::Preprocess(const cv::Mat& img) {
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
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample.convertTo(sample_float,CV_32FC3);
  else
    sample.convertTo(sample_float,CV_32FC1);
  cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);

  return sample_float;
}

void mtcnnDetector::WrapInputLayer(std::vector<cv::Mat>* input_channels,
        Blob<float>* input_layer, const int height, const int width) {
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void mtcnnDetector::ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_float,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName){
  int numBox = regressed_rects.size();
  Blob<float>* crop_input_layer = net->input_blobs()[0];
  int input_channels = crop_input_layer->channels();
  int input_width  = crop_input_layer->width();
  int input_height = crop_input_layer->height();
  crop_input_layer->Reshape(1, input_channels, input_width, input_height);
  net->Reshape();

  condidate_rects_.clear();
  // load crop_img data to datum
  for(int i=0;i<numBox;i++){
    std::vector<cv::Mat> channels;
    WrapInputLayer(&channels,net->input_blobs()[0],input_width,input_height);
    int pad_top   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_left  = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
    int pad_bottom= std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

    cv::Mat crop_img = sample_float(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_left,pad_right,pad_top,pad_bottom,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height));
    crop_img = (crop_img-127.5)*0.0078125;
    cv::split(crop_img,channels);

    CHECK(reinterpret_cast<float*>(channels.at(0).data) == net->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
    net->Forward();

    int reg_id = 0;
    int confidence_id = 1;
    if(netName == 'o') confidence_id = 2;
    const Blob<float>* reg = net->output_blobs()[reg_id];
    const Blob<float>* confidence = net->output_blobs()[confidence_id];
    // ONet points_offset != NULL
    const Blob<float>* points_offset = net->output_blobs()[1];

    const float* confidence_data = confidence->cpu_data() + confidence->count()/2;
    const float* reg_data = reg->cpu_data();
    const float* points_data;
    if(netName == 'o') points_data = points_offset->cpu_data();

    if(*(confidence_data) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[0],reg_data[1],reg_data[2],reg_data[3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float w = faceRect.y2 - faceRect.y1 + 1;
        float h = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j+5) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
  regressed_pading_.clear();
}

// multi test image pass a forward
void mtcnnDetector::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_float,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName){
  condidate_rects_.clear();

  int numBox = regressed_rects.size();
  std::vector<Datum> datum_vector;

  boost::shared_ptr<MemoryDataLayer<float> > mem_data_layer;
  mem_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layers()[0]);
  int input_width  = mem_data_layer->width();
  int input_height = mem_data_layer->height();

  // load crop_img data to datum
  for(int i=0;i<numBox;i++){
    int pad_left   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_top  = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_bottom = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
    int pad_right= std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

    cv::Mat crop_img = sample_float(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_top,pad_bottom,pad_left,pad_right,cv::BORDER_CONSTANT,cv::Scalar(0));

    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height));
    crop_img = (crop_img-127.5)*0.0078125;
    Datum datum;
    CvMatToDatumSignalChannel(crop_img,&datum);
    //CVMatToDatum(crop_img,&datum);
    datum_vector.push_back(datum);
  }
  regressed_pading_.clear();

  /* extract the features and store */
  mem_data_layer->set_batch_size(numBox);
  mem_data_layer->AddDatumVector(datum_vector);
  /* fire the network */
  float no_use_loss = 0;
  net->Forward(&no_use_loss);
//  CHECK(reinterpret_cast<float*>(crop_img_set.at(0).data) == net->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";

  // return RNet/ONet result
  std::string outPutLayerName = (netName == 'r' ? "conv5-2" : "conv6-2");
  std::string pointsLayerName = "conv6-3";

  const boost::shared_ptr<Blob<float> > reg = net->blob_by_name(outPutLayerName);
  const float* reg_data = reg->cpu_data();

  const boost::shared_ptr<Blob<float> > confidence = net->blob_by_name("prob1");
  const float* confidence_data = confidence->cpu_data();

  const float* points_data = NULL;
  // ONet points_offset != NULL
  if(netName == 'o') {
    const boost::shared_ptr<Blob<float> > points_offset = net->blob_by_name(pointsLayerName);
    points_data = points_offset->cpu_data();
  }

  for(int i=0;i<numBox;i++){
    if(*(confidence_data+i*2+1) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data+i*2+1);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[4*i+0],reg_data[4*i+1],reg_data[4*i+2],reg_data[4*i+3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float w = faceRect.y2 - faceRect.y1 + 1;
        float h = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j+10*i) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j+5+10*i) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
}

bool mtcnnDetector::CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum){
  if (cv_mat.empty())
    return false;
  int channels = cv_mat.channels();

  datum->set_channels(cv_mat.channels());
  datum->set_height(cv_mat.rows);
  datum->set_width(cv_mat.cols);
  datum->set_label(0);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);

  int datum_height = datum->height();
  int datum_width  = datum->width();
  if(channels == 3){
    for(int c = 0;c < channels;c++){
      for (int h = 0; h < datum_height; ++h){
        for (int w = 0; w < datum_width; ++w){
          const float* ptr = cv_mat.ptr<float>(h);
          datum->add_float_data(ptr[w*channels+c]);
        }
      }
    }
  }

  return true;
}

bool ssdDetector::init(NYUnpacker &unp, const string& modelFilePath, const string& trainedFilePath, int device) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::SetDevice(device);
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  //net_.reset(new Net<float>(modelFilePath, TEST));
  //net_->CopyTrainedLayersFrom(trainedFilePath);
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

  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  /* Load the binaryproto mean file. */
  float mean_value[3] = {104,117,123};
  SetMean(mean_value);

  is_inited_=true;
  return is_inited_;
}

void ssdDetector::prealloc(int times) {
  if (!is_inited_) {
    std::cout << "ssdDetector Not Inited." << std::endl;
    return;
  }
  cv::Mat t;
  t = cv::Mat::zeros(100,100,CV_32FC3);
  detect(t);
}

std::vector<FaceInfo> ssdDetector::detect(const cv::Mat& img, float threshold) {
  if (!is_inited_) {
    std::cout << "ssdDetector Not Inited." << std::endl;
    std::vector<FaceInfo> t;
    return t;
  }
#ifndef NDEBUG
  double t1 = getTime();
#endif
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  cv::Mat sample_normalized;
  sample_normalized = Preprocess(img);

  cv::split(sample_normalized, input_channels);

  CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  int num_det = output_layer->height();
  std::vector<FaceInfo> faceInfos;
  for (int k = 0; k < num_det; ++k) {
    const float* result = output_layer->cpu_data() + output_layer->offset(0,0,k);
    if (result[0] == -1) {
      continue;
    } else if (result[2] <= threshold) {
      continue;
    }
    FaceInfo faceInfo;
    faceInfo.bbox.x1 = result[3] * img.cols;
    faceInfo.bbox.y1 = result[4] * img.rows;
    faceInfo.bbox.x2 = result[5] * img.cols;
    faceInfo.bbox.y2 = result[6] * img.rows;
    faceInfo.bbox.score = result[2];
    faceInfos.push_back(faceInfo);
  }
  sort(faceInfos.begin(), faceInfos.end(), cmpFunc(AREA));
#ifndef NDEBUG
  double t2 = getTime();
  std::cout << "Time of ssdDetector: " << t2-t1 << std::endl;
#endif
  return faceInfos;
}

/* Load the mean file in binaryproto format. */
void ssdDetector::SetMean(const float mean_value[]) {
    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(mean_value[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void ssdDetector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

cv::Mat ssdDetector::Preprocess(const cv::Mat& img) {
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

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  return sample_normalized;
}
