/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: track/KalmanTracker.h
 *     Created: 2017-11-07 16:47
 * Description:
 * ===================================================
 */
///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H

#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
    KalmanTracker()
    {
        cout<<"an new initialize happen,no id"<<endl;
        init_kf(StateType());
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = 0;

        //kf_count++;
    }
    KalmanTracker(StateType initRect, int id)
    {


        /*StateType stateMat=initRect;
        int stateNum = 7;
        int measureNum = 4;
        cout<<"test"<<endl;
        kf = KalmanFilter(stateNum, measureNum, 0);
        cout<<"test1"<<endl;
        measurement = Mat::zeros(measureNum, 1, CV_32F);
        cout<<"2"<<endl;
        float data[50] ={
                        1, 0, 0, 0, 1, 0, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 0, 1, 0, 0, 0, 1,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 1
                        };
        cout << "ck" << endl;
        cv::Mat tmp = cv::Mat(stateNum, stateNum, CV_32F, data);

        kf.transitionMatrix = tmp.clone();
        kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1);

        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(kf.errorCovPost, Scalar::all(1));

        // initialize state vector with bounding box in [cx,cy,s,r] style
        kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
        kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
        kf.statePost.at<float>(2, 0) = stateMat.area();
        kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
        cout<<"you"<<endl;*/
        if(id==58){
            cout<<"test"<<endl;
        }
        init_kf(initRect);

        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = id;
        //kf_count++;
    }

    ~KalmanTracker()
    {
        m_history.clear();
    }

    StateType predict();
    void update(StateType stateMat);

    StateType get_state();
    StateType get_rect_xysr(float cx, float cy, float s, float r);


    int getId();
    void setId(int i);
    //static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

    cv::KalmanFilter kf;
    cv::Mat measurement;
private:
    void init_kf(StateType stateMat);

   // cv::KalmanFilter kf;
    //cv::Mat measurement;

    std::vector<StateType> m_history;
};




#endif
