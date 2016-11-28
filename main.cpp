#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#include <lmdb.h>
#include <random>
#include <algorithm>

#include "src/proto/test.pb.h"
#include "src/json/json/json.h"

constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;
constexpr int HALF_WIDTH = WIDTH / 2;
constexpr int HALF_HEIGHT = HEIGHT / 2;

using namespace cv;
using std::string;
using std::cout;
using std::endl;
using std::set;
using std::vector;

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

class ImageInfo {
public:
    ImageInfo(Json::Value &value) {
        string fileName = value["file"].asString();
        rect.x = value["left"].asInt();
        rect.y = value["top"].asInt();
        rect.width = value["right"].asInt() - rect.x;
        rect.height = value["bottom"].asInt() - rect.y;
        img = imread(fileName);
    }

    Mat subImg(int x, int y, int w, int h) {
        return img(cv::Rect(x, y, w, h));
    }

    Rect rect;
    Mat img;

};

void saveImage(Json::Value &value, std::map<string, int> & map, MDB_txn *txn, MDB_dbi dbi, MDB_env *env, int &counter);

void saveType(Json::Value &value, std::map<string, int> & labelMap, ImageInfo info, set<ImgPoint> &points,
              vector<string> &matrixes);

void save(const string &dataToWrite, MDB_txn *txn, MDB_dbi &dbi, int &counter);
string serialize(Mat &subImg, int32_t label);

std::vector<ImgPoint> getPoints(Json::Value &value);


int main(int argc, char **argv) {
    std::map<string, int> mapType;

    if (argc != 3) {
        std::cerr << "need data file and output file" << std::endl;
        exit(-1);
    }

    std::ifstream iStream(argv[1]);
    Json::Value root;

    iStream >> root;

    MDB_env *env= nullptr;
    MDB_env *env_test= nullptr;

    struct MDB_envinfo current_info;

    MDB_dbi dbi;
    MDB_dbi dbi_test;
    MDB_txn *txn= nullptr;
    MDB_txn *txn_test= nullptr;
    int counter = 0;
    int counterTest=0;

    mdb_env_create(&env);

    mdb_env_set_mapsize(env, 1000*1024*1024);

    mdb_env_open(env, argv[2], 0, 0664);
    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_open(txn, NULL, 0, &dbi);

    mdb_env_create(&env_test);
    mdb_env_set_mapsize(env_test, 100*1024*1024);

    mdb_env_open(env_test, (argv[2]+string(".test")).c_str(), 0, 0664);
    mdb_txn_begin(env_test, NULL, 0, &txn_test);
    mdb_open(txn_test, NULL, 0, &dbi_test);



    for (auto &child: root) {
        if (!child["test"].asInt() ){
            cout << "data image from " << child["file"] << endl;
            saveImage(child, mapType, txn, dbi, env, counter);
        } else {
            cout << "Test data image from " << child["file"] << endl;
            saveImage(child, mapType, txn_test, dbi_test, env_test, counterTest);
            }
    }
    mdb_txn_commit(txn  );
    mdb_txn_commit(txn_test);
    mdb_close(env, dbi);
    mdb_close(env_test, dbi_test);
    std::cout << mapType.size() << " labels" << std::endl;
    std::cout << counter << " images " << std::endl;
    std::cout << counterTest << " test images " << std::endl;

    return 0;
}

void
saveImage(Json::Value &value, std::map<string, int> & mapType, MDB_txn *txn, MDB_dbi dbi, MDB_env *env, int &counter) {
    ImageInfo imageInfo(value);
    std::set<ImgPoint> points;
    std::vector<string> matrixes;


    auto types = value["type"];
    for (auto &type: types) {
        saveType(type, mapType, imageInfo, points, matrixes);
    }

    int left = value["left"].asInt();
    int right = value["right"].asInt();
    int top = value["top"].asInt();
    int bottom = value["bottom"].asInt();

    int numValidImages = matrixes.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDis(left, right - WIDTH - 1);
    std::uniform_int_distribution<> yDis(top, bottom - HEIGHT - 1);

    for (int i = 0; i < numValidImages * 4; i++) {

        ImgPoint point(xDis(gen), yDis(gen));
        while (points.count(point) > 0) {
            point = ImgPoint(xDis(gen), yDis(gen));
        }
        Mat subImg = imageInfo.subImg(point.x, point.y, WIDTH, HEIGHT);
        matrixes.push_back(serialize(subImg, 0));
    }


    std::random_shuffle(std::begin(matrixes), std::end(matrixes));

    for (const auto &matrix:matrixes) {
        save(matrix, txn, dbi, counter);
        if (counter % 100 == 99) {
            mdb_txn_commit(txn);
            mdb_txn_begin(env, NULL, 0, &txn);
        }
    }
}


string serialize(Mat &subImg, int32_t label) {
    genSamples::Datum datum;
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
    string out;
    datum.SerializeToString(&out);
    return out;
}

void save(const string &dataToWrite, MDB_txn *txn, MDB_dbi &dbi, int &counter) {
    MDB_val key, data;
    key.mv_size = sizeof(int);
    key.mv_data = &counter;
    data.mv_size = dataToWrite.size();
    data.mv_data = const_cast<char *>(dataToWrite.c_str());
    mdb_put(txn, dbi, &key, &data, 0);
    counter++;
}

void saveType(Json::Value &type, std::map<string, int> & labelMap, ImageInfo info, set<ImgPoint> &pointsSet,
              std::vector<string> &matrixes) {
    auto label = type.begin().key().asString();
    auto points = getPoints(*type.begin());
    cout << "labelSize: " << labelMap.size() << endl;
    if (labelMap.count(label) == 0) {
        labelMap[label] = labelMap.size() + 1;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDis(0, WIDTH - 1);
    std::uniform_int_distribution<> yDis(0, HEIGHT - 1);
    std::uniform_int_distribution<> colDis(0, 3);
    std::uniform_int_distribution<> deltaDis(-10, 10);

    int labelIndex = labelMap[label];
    for (auto &point: points) {
        pointsSet.insert(point);
        Mat subImg = info.subImg(point.x - HALF_WIDTH, point.y - HALF_HEIGHT, WIDTH, HEIGHT);
        matrixes.push_back(serialize(subImg, labelIndex));
        for (int i = 0; i < 40; i++) {
            Mat cloned = subImg.clone();
            for (int i = 0; i < 30; i++) {
                int y = yDis(gen);
                int x = xDis(gen);
                int colIndex = colDis(gen);
                int delta = deltaDis(gen);
                int col = cloned.at<cv::Vec3b>(y, x)[colIndex];
                if (col + delta < 0) {
                    col = 0;
                } else if (col + delta > 255) {
                    col = 255;
                } else {
                    col += delta;
                }
                cloned.at<cv::Vec3b>(y, x)[colIndex] = col;
            }
            matrixes.push_back(serialize(cloned, labelIndex));
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
