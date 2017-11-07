/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: track/Track.cpp
 *     Created: 2017-11-07 16:47
 * Description:
 * ===================================================
 */
#include "Track.h"
#include "KalmanTracker.h"


using namespace std;
using namespace cv;


Track::Track(int f)
{
    MaxFailnumber = f;
    CurIDNumber = 0;
    nameIdx=1;
}

void Track::InitiaTracker(const std::vector<FaceInfo> &faceInfo, std::vector<long long int> &ID)
{
    for(int i = 0; i < (int)faceInfo.size(); i++)
    {
        float x1 = faceInfo[i].bbox.x1;
        float y1 = faceInfo[i].bbox.y1;
        float x2 = faceInfo[i].bbox.x2;
        float y2 = faceInfo[i].bbox.y2;

        float w = x2 - x1 + 1;
        float h = y2 - y1 + 1;

        KalmanTracker tmp_kt(Rect(x1, y1, w, h), CurIDNumber);

        KTracker trk = {tmp_kt, 0};

        trackers.push_back(trk);
        ID.push_back(CurIDNumber);
        CurIDNumber++;
    }
}

//get the det box according the faceInfo
void Track::get_DetBox(const std::vector<FaceInfo> &faceInfo)
{
    //clear det box
    if(!Det.empty())
    {
        vector<Box>().swap(Det);
    }

    //get the det Box according to the faceInfo withou id
    for(int i = 0; i < (int)faceInfo.size(); i++)
    {
        float x1 = faceInfo[i].bbox.x1;
        float y1 = faceInfo[i].bbox.y1;
        float x2 = faceInfo[i].bbox.x2;
        float y2 = faceInfo[i].bbox.y2;
        float w = x2 - x1 + 1;
        float h = y2 - y1 + 1;

        //create the Det vec withou id,false match
        Box tmp = {cv::Rect(x1, y1, w, h),  -1};
        Det.push_back(tmp);
    }
}


float Track::caculateIOU(const cv::Rect &rectA, const cv::Rect &rectB)
{
    if (rectA.x > rectB.x + rectB.width)
    {
        return 0.;
    }
    if (rectA.y > rectB.y + rectB.height)
    {
        return 0.;
    }
    if ((rectA.x + rectA.width) < rectB.x)
    {
        return 0.;
    }
    if ((rectA.y + rectA.height) < rectB.y)
    {
        return 0.;
    }


    float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
    float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
    float intersection = colInt * rowInt;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float intersectionPercent = intersection / (areaA + areaB - intersection);


    return intersectionPercent;
}


//search for a corresponding box and get the track box using motion model
void Track::Search(int index)
{
    int trackId = -1;
    int pre_id = -1;
    double score = threshold;
    //double max = -1;

    //int i = 7;


    for(int i = 0; i < (int)Tra.size(); i++)
    {
        //hasn't been marked yet
        if(!trackers[i].Match)
        {
            double score_tmp = caculateIOU(Det[index].box, Tra[i].box);
            if(score_tmp > score)
            {
                trackId = Tra[i].id;
                score = score_tmp;
                pre_id = i;
            }
        }
    }


    //find the correspoind box
    if(trackId >= 0)
    {

        trackers[pre_id].Match = true;
        Det[index].id = trackId;
    }
    // not find the correspondin box so a new tracker
    else
    {
        KTracker trk = {KalmanTracker(Det[index].box, CurIDNumber), false, 0};
        Det[index].id = CurIDNumber;
        trackers.push_back(trk);
        CurIDNumber++;


    }

    return ;
}

void Track::Associate()
{
    for(int i = 0; i < (int)Det.size(); i++)
    {
        Search(i);
    }
    for(int i = 0; i < (int)Tra.size(); i++)
    {
        if(!trackers[i].Match)  //can not find a corresponding box in the next fram
        {

            if(trackers[i].failnumber < MaxFailnumber)  //still have a chance
            {
                trackers[i].failnumber++;
            }
            else
            {
                trackers.erase(trackers.begin() + i); //delete the tracker
            }
        }
    }

}



void Track::getID(const std::vector<FaceInfo> &faceInfo, float thr, std::vector<long long int> &ID)
{
    threshold = thr;

    //Initialize the trakers at the beginning
    if(trackers.size() == 0)
    {
        InitiaTracker(faceInfo, ID);
        return ;
    }

    if(!ID.empty()){
        cout<<"Input ID is not empty when you call Track::getID"<<endl;
        exit(1);
    }

    get_DetBox(faceInfo);

    //initialize Match flag agagin to associate
    for(int i = 0; i < (int)trackers.size(); i++)
    {
        trackers[i].Match = false;
    }

    //clear the Tra box if not empty
    if(!Tra.empty())
    {
        vector<Box>().swap(Tra);
    }

    //get predicted locations from existing trackers.
    for (vector<KTracker>::iterator it = trackers.begin(); it != trackers.end();)
    {
        Rect_<float> pBox = (*it).kt.predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            Box tmp = {pBox, (*it).kt.getId()};
            Tra.push_back(tmp);
            it++;
        }
        else
        {
            //release the tracker not in the frame
            it = trackers.erase(it);
        }
    }

    //associate the track box and the detect box
    Associate();

    //get the associate pair
    vector<MatchPair> matchP;
    for(int i = 0; i < (int)trackers.size(); i++)
    {
        for(int j = 0; j < (int)Det.size(); j++)
        {
            if(trackers[i].Match==true&&trackers[i].kt.getId() >= 0 && trackers[i].kt.getId() == Det[j].id)  //tracker is not the beginner and correspoinding
            {
                MatchPair m = {j, i};
                matchP.push_back(m);
            }
        }
    }

    //update the tracker
    for (unsigned int i = 0; i < matchP.size(); i++)
    {
        trackers[matchP[i].TraId].kt.update(Det[matchP[i].DetId].box);
    }

    //output the result
    for(int i=0;i<(int)Det.size();i++){
        ID.push_back(Det[i].id);
    }
}
