//
// Created by paolo on 29/11/16.
//

#ifndef NEURALNETWORKCAFFETEST_TESTIMAGES_H
#define NEURALNETWORKCAFFETEST_TESTIMAGES_H

#include <vector>
#include <tuple>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/opencv.hpp>

#include "./proto/test.pb.h"
#include "./json/json/json.h"

class TestImages {
    class ImgPoint {
    public:
        ImgPoint(Json::Value &value) {
            x = std::stoi(value["x"].asString());
            y = std::stoi(value["y"].asString());
        }

        ImgPoint(int x, int y) : x{x}, y{y} {}

        bool operator==(const ImgPoint &point) const {
            return x == point.x && y == point.y;
        }

        bool operator<(const ImgPoint &point) const {
            return (x == point.x) ? (y < point.y) : (x < point.x);
        }

        int x;
        int y;
    };

public:
    TestImages(Json::Value & rootImage);

    static int WIDTH;
    static int HEIGHT;

    size_t size() {return background.size();}
    std::tuple<caffe::Datum, cv::Mat> get(int index){return background[index];}
private:
    caffe::Datum serialize(cv::Mat &subImg, int label);
    std::vector<ImgPoint> getPoints(Json::Value &value);
    std::vector<std::tuple<caffe::Datum, cv::Mat>> background;
    std::set<ImgPoint> points;
    int left;
    int right;
    int top;
    int bottom;
    std::string filename;
};


#endif //NEURALNETWORKCAFFETEST_TESTIMAGES_H
