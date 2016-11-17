#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "src/proto/test.pb.h"
#include "src/json/json/json.h"

constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;
constexpr int HALF_WIDTH = WIDTH / 2;
constexpr int HALF_HEIGHT = HEIGHT / 2;

using namespace cv;
using std::string;


class ImgPoint {
public:
    ImgPoint(Json::Value &value) {
        x = std::stoi(value["x"].asString());
        y = std::stoi(value["y"].asString());
    }

    int x;
    int y;
};

class ImageInfo {
public:
    ImageInfo(Json::Value &value) {
        string fileName = value["file"].asString();
        rect.x = value["left"].asInt();
        rect.y = value["top"].asInt();
        rect.width = value["right"].asInt()-rect.x;
        rect.height= value["bottom"].asInt()-rect.y;
        img = imread(fileName);
    }

    Mat subImg(int x, int y, int w, int h){
        return img( cv::Rect(x,y,w,h));
    }

    Rect rect;
    Mat img;

};

void saveImage(Json::Value &value, std::map<string, int> &map);

void saveType(Json::Value &value, std::map<string, int> &labelMap, ImageInfo &info);

std::vector<ImgPoint> getPoints(Json::Value &value);




int main(int argc, char **argv) {
    std::map<string, int> mapType;

    if (argc != 2) {
        std::cerr << "need data file" << std::endl;
        exit(-1);
    }

    std::ifstream iStream(argv[1]);
    Json::Value root;

    iStream >> root;

    for (auto &child: root) {
        saveImage(child, mapType);
    }

    return 0;
}

void saveImage(Json::Value &value, std::map<string, int> &mapType) {
    ImageInfo imageInfo(value);

    auto types = value["type"];
    for (auto &type: types) {
        saveType(type, mapType, imageInfo);
    }
}

void saveType(Json::Value &type, std::map<string, int> &labelMap, ImageInfo &info) {
    genSamples::Datum datum;
    auto label = type.begin().key().asString();
    auto points = getPoints(*type.begin());
    if (labelMap.count(label) == 0) {
        labelMap[label] = labelMap.size() + 1;
    }
    datum.set_channels(3);
    datum.set_height(HEIGHT);
    datum.set_width(WIDTH);
    datum.set_label(labelMap[label]);
    for (auto &point: points) {
        Mat subImg = info.subImg(point.x, point.y, WIDTH, HEIGHT);
        std::cout << subImg.size() << std::endl;
        std::vector<uint8_t > vectorData;
        for (MatIterator_<uint8_t > iter = subImg.begin(); iter != subImg.end(); iter++){
            vectorData.push_back(*iter);
        }
        std::cout << vectorData.size() << std::endl;
        datum.set_data(subImg.data, subImg.cols * subImg.rows);
    }
}

std::vector<ImgPoint> getPoints(Json::Value &value) {
    std::vector<ImgPoint> result;
    for (auto &point: value) {
        result.emplace_back(point);
    }
    return result;
}