//
// Created by paolo on 29/11/16.
//
#include <opencv2/opencv.hpp>
#include "TestImages.h"


using namespace cv;
using std::string;


int TestImages::WIDTH=16;
int TestImages::HEIGHT=16;


TestImages::TestImages(Json::Value & rootImage) {
    left = rootImage["left"].asInt();
    right = rootImage["right"].asInt();
    top = rootImage["top"].asInt();
    bottom = rootImage["bottom"].asInt();

    filename = rootImage["file"].asString();

    std::cout << "Reading " << filename << std::endl;

    Mat img = imread(filename);

    auto types = rootImage["type"];
    for (auto &type: types) {
        auto newPoints = getPoints(*type.begin());
        points.insert(newPoints.begin(), newPoints.end());
    }

    for (int x=left; x < right-WIDTH; x+=2){
        for (int y=top; y < bottom-HEIGHT; y+=2){
          //  if (points.count({x,y})==0) {
                auto subImg = img(cv::Rect(x, y, WIDTH, HEIGHT));
                background.emplace_back(serialize(subImg,0), subImg);
         //   }
        }
    }

}

std::vector<TestImages::ImgPoint> TestImages::getPoints(Json::Value &value) {
    std::vector<ImgPoint> result;
    for (auto &point: value) {
        result.emplace_back(point);
    }
    return result;
}



caffe::Datum TestImages::serialize(Mat & subImg,int label) {
    caffe::Datum datum;
    datum.set_channels(3);
    datum.set_height(HEIGHT);
    datum.set_width(WIDTH);
    datum.set_label(label);

    std::unique_ptr<uint8_t> data{new uint8_t[WIDTH * HEIGHT * 3]};
    uint8_t *dataIter;
    dataIter = data.get();
    for (auto iter = subImg.begin<Vec3b>(); iter != subImg.end<Vec3b>(); iter++) {
        *dataIter = (*iter).val[0];
        dataIter++;
        *dataIter = (*iter).val[1];
        dataIter++;
        *dataIter = (*iter).val[2];
        dataIter++;
    }
    datum.set_data(data.get(), WIDTH * HEIGHT * 3);
    return datum;
}