/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: track/Track.h
 *     Created: 2017-11-07 16:47
 * Description:
 * ===================================================
 */
#ifndef TRACK_H
#define TRACK_H

#include <opencv2/opencv.hpp>
#include <faceinfo.hpp>
#include "time.h"
#include <sys/time.h>
#include <string>
#include "KalmanTracker.h"


typedef struct Box
{
    cv::Rect box;
    long long int id;
} Box;

typedef struct KTracker
{
    KalmanTracker kt;
    bool Match;
    int failnumber;
} KTracker;


typedef struct MatchPair
{
    int DetId;
    int TraId;
} MatchPair;


class Track
{
public:
    Track(int f);            //t:threshold, u:MaxFailnumber
    void getID(const std::vector<FaceInfo> &faceInfo, float threshold, std::vector<long long int> &ID); // input the detec faceinfor, Output the correspoind ID

private:
    void InitiaTracker(const std::vector<FaceInfo> &faceInfo, std::vector<long long int> &ID);              //initialize the trackers
    void get_DetBox(const std::vector<FaceInfo> &faceInfo);         //transform the faceinfo to the proper structure
    void Associate();               //Match the det and track box
    void Search(int index);         //find the corresponding track box for the det box
    float caculateIOU(const cv::Rect &rectA, const cv::Rect &rectB);

private:
    long long int CurIDNumber;
    int MaxFailnumber;
    double threshold;

    int nameIdx;
    //  Kalman tracker
    std::vector<KTracker> trackers;
    // std::vector<int> &ID;

    std::vector<Box> Det;   //det box from the dector

    std::vector<Box> Tra;   //track box according the trackers predict
};





#endif
