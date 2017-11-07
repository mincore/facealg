/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: track.cpp
 *     Created: 2017-07-19 14:34
 * Description:
 * ===================================================
 */
#include "track/Track.h"

#define API extern "C"

API void* NYTrackNew(int maxLoss)
{
    return new Track(maxLoss);
}

API void NYTrack(void*h, const std::vector<FaceInfo> &faceInfo, float threshold, std::vector<long long> &trackIDs)
{
    ((Track*)h)->getID(faceInfo, threshold, trackIDs);
}

API void NYTrackDel(void*h)
{
    delete (Track*)h;
}


