#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#include <lmdb.h>
#include <random>

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
        rect.width = value["right"].asInt() - rect.x;
        rect.height = value["bottom"].asInt() - rect.y;
        img = imread(fileName);
        img.convertTo(img, CV_32F);
    }

    Mat subImg(int x, int y, int w, int h) {
        return img(cv::Rect(x, y, w, h));
    }

    Rect rect;
    Mat img;

};

void saveImage(Json::Value &value, std::map<string, int> &map, MDB_txn *env, MDB_dbi &dbi, int &i);

void saveType(Json::Value &value, std::map<string, int> &labelMap, ImageInfo info, MDB_txn *txn, MDB_dbi &dbi,
              int &counter);

std::vector<ImgPoint> getPoints(Json::Value &value);

std::string getImageType(int number);


int main(int argc, char **argv) {
    std::map<string, int> mapType;

    if (argc != 3) {
        std::cerr << "need data file and output file" << std::endl;
        exit(-1);
    }

    std::ifstream iStream(argv[1]);
    Json::Value root;

    iStream >> root;

    MDB_env *env;
    mdb_env_create(&env);
    mdb_env_open(env, argv[2], 0, 0664);
    MDB_dbi dbi;
    MDB_txn *txn;
    mdb_txn_begin(env, NULL, 0, &txn);
    int counter = 0;
    mdb_open(txn, NULL, 0, &dbi);

    for (auto &child: root) {
        saveImage(child, mapType, txn, dbi, counter);
    }
    mdb_txn_commit(txn);
    std::cout << mapType.size() << " labels" << std::endl;
    std::cout << counter << " images " << std::endl;


//    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
//    dbi = lmdb::dbi::open(rtxn, nullptr);
//    auto cursor = lmdb::cursor::open(rtxn, dbi);
//    std::string key, value;
//    while (cursor.get(key, value, MDB_NEXT)) {
//        genSamples::Datum  datum;
//        datum.ParseFromString(value);
//        std::cout << datum.channels() << std::endl;
//    }
//    cursor.close();

    return 0;
}

void saveImage(Json::Value &value, std::map<string, int> &mapType, MDB_txn *txn, MDB_dbi &dbi, int &counter) {
    ImageInfo imageInfo(value);


    auto types = value["type"];
    for (auto &type: types) {
        saveType(type, mapType, imageInfo, txn, dbi, counter);
    }
    // txn.reset();
}


string serialize(Mat &subImg, int32_t label) {
    genSamples::Datum datum;
    datum.set_channels(3);
    datum.set_height(HEIGHT);
    datum.set_width(WIDTH);
    datum.set_label(label);

    std::unique_ptr<uint8_t> data(new uint8_t[WIDTH * HEIGHT * 3]);
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
    string out;
    datum.SerializeToString(&out);
    return out;
}

void save(string &dataToWrite, MDB_txn *txn, MDB_dbi &dbi, int &counter) {
    MDB_val key, data;
    key.mv_size = sizeof(int);
    key.mv_data = &counter;
    data.mv_size = dataToWrite.size();
    data.mv_data = const_cast<char *>(dataToWrite.c_str());
    mdb_put(txn, dbi, &key, &data, 0);
    counter++;
}

void saveType(Json::Value &type, std::map<string, int> &labelMap, ImageInfo info, MDB_txn *txn, MDB_dbi &dbi,
              int &counter) {
    auto label = type.begin().key().asString();
    auto points = getPoints(*type.begin());
    if (labelMap.count(label) == 0) {
        labelMap[label] = labelMap.size() + 1;
    }

    std::uniform_int_distribution<> xDis(0, WIDTH);
    std::uniform_int_distribution<> yDis(0, HEIGHT);
    std::uniform_int_distribution<> colDis(0, 3);
    std::uniform_int_distribution<> delta(-10, 10);


    for (auto &point: points) {
        Mat subImg = info.subImg(point.x - HALF_WIDTH, point.y - HALF_HEIGHT, WIDTH, HEIGHT);
        string out{serialize(subImg, labelMap[label])};
        save(out, txn, dbi, counter);
        for (int i=0; i < 30; i++){
            Mat cloned = subImg.clone();
            for (int i = 0; i < 20;i++){
                cloned.at<Vec3b>(yDis,xDis);
            }
        }
    }

}

std::vector<ImgPoint> getPoints(Json::Value &value) {
    std::vector<ImgPoint> result;
    for (auto &point: value) {
        result.emplace_back(point);
    }
    return result;
}

std::string getImageType(int number) {
    // find type
    int imgTypeInt = number % 8;
    std::string imgTypeString;

    switch (imgTypeInt) {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number / 8) + 1;

    std::stringstream type;
    type << "CV_" << imgTypeString << "C" << channel;

    return type.str();
}