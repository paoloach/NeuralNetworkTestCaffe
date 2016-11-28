//
// Created by paolo on 25/11/16.
//

#include <vector>
#include <string>
#include <iostream>
#include <lmdb.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "./proto/test.pb.h"
#include "./json/json/json.h"

constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;

using std::string;
using std::vector;
using std::cout;
using std::endl;
using namespace cv;

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

vector<string> typeImagesPath = {"images/circleGreen.png", "images/squareYellow.png", "images/circleGreen.png",
                                 "images/squarePurple.png", "images/circleBlue.png",};
vector<string> background = {"images/background1.png",
                             "images/background2.png",
                             "images/background3.png",
                             "images/background4.png",
                             "images/background5.png",
                             "images/background6.png",
                             "images/background7.png",
                             "images/background8.png",
                             "images/background9.png",
                             "images/background10.png",
                             "images/background11.png"};

Mat addBackround(Mat &mat, Mat &mat1);

void add(Mat &bg, Mat &mat, int left, int top);

string serialize(Mat &&subImg, int32_t label);
void save(const string &dataToWrite, MDB_txn *txn, MDB_dbi &dbi, int &counter);

void addBackground(Json::Value &value, vector<string> & data) ;
void getPoints(Json::Value &value, std::set<ImgPoint> & points) ;

int main(int argc, char **argv) {
    vector<Mat> images;
    vector<Mat> imagesBackground;
    vector<string> data;


    if (argc != 3) {
        std::cerr << "need output directory" << std::endl;
        exit(-1);
    }


    std::ifstream iStream(argv[2]);
    Json::Value root;

    iStream >> root;

    MDB_env *env = nullptr;

    struct MDB_envinfo current_info;

    MDB_dbi dbi;
    MDB_txn *txn = nullptr;
    int counter = 0;

    mdb_env_create(&env);
    mdb_env_set_mapsize(env, 1000*1024*1024);

    mdb_env_open(env, argv[1], 0, 0664);
    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_open(txn, NULL, 0, &dbi);

    for (const auto &image : typeImagesPath) {
        images.push_back(imread(image, CV_LOAD_IMAGE_UNCHANGED));
    }
    for (const auto &image : background) {
        imagesBackground.push_back(imread(image, CV_LOAD_IMAGE_UNCHANGED));
    }

    int type = 1;
    for (auto &image: images) {
        for (auto &imageBG: imagesBackground) {
            data.push_back(serialize(std::move(addBackround(image, imageBG)), type));
        }
        type++;
    }

    type=1;
    for (auto &image: images) {
        for (auto &other: images) {
            if (&image != &other) {
                for (auto &imageBG: imagesBackground) {
                    data.push_back(serialize(std::move(addBackround(image, imageBG)), type));
                    for (int x = 1; x < WIDTH; x++) {
                        for (int y = 1; y < HEIGHT; y++) {
                            auto bg = imageBG.clone();
                            add(bg, image, 0, 0);
                            add(bg, other, x,y);
                            data.push_back(serialize(std::move(bg(cv::Rect(0,0,WIDTH,HEIGHT))), type));
                            bg = imageBG.clone();
                            add(bg, image, WIDTH, 0);
                            add(bg, other, x+2,y+2);
                            data.push_back(serialize(std::move(bg(cv::Rect(0,0,WIDTH,HEIGHT))), type));
                            bg = imageBG.clone();
                            add(bg, image, 0, HEIGHT);
                            add(bg, other, x,y);
                            data.push_back(serialize(std::move(bg(cv::Rect(0,0,WIDTH,HEIGHT))), type));
                            bg = imageBG.clone();
                            add(bg, image, WIDTH, HEIGHT);
                            add(bg, other, x+2,y+2);
                            data.push_back(serialize(std::move(bg(cv::Rect(0,0,WIDTH,HEIGHT))), type));
                        }
                    }
                }
            }
        }
        type++;
    }

    for (auto &child: root) {
        if (!child["test"].asInt() ){
            cout << "data image from " << child["file"] << endl;
            addBackground(child, data);
        }
    }

    cout << "created " << data.size() << " samples" << endl;
    cout << "now shuffling" << endl;
    std::random_shuffle(std::begin(data), std::end(data));
    cout << "now saving" << endl;
    counter = 0;
    for (const auto &matrix:data) {
        save(matrix, txn, dbi, counter);
        if (counter % 100 == 99) {
            mdb_txn_commit(txn);
            mdb_txn_begin(env, NULL, 0, &txn);
        }
    }
    cout << "finished" << endl;
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

Mat addBackround(Mat &mat, Mat &background) {
    Mat result = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            auto pixel = mat.at<Vec4b>(row, col);
            if (pixel.val[3] == 0) {
                pixel = background.at<Vec4b>(row, col);
            }
            Vec3b resultPixel;
            resultPixel.val[0] = pixel.val[0];
            resultPixel.val[1] = pixel.val[1];
            resultPixel.val[2] = pixel.val[2];
            result.at<Vec3b>(cv::Point(row, col)) = resultPixel;
        }
    }

    return result;
}

void add(Mat &bg, Mat &mat, int left, int top) {
    int x, y;
    int xm, ym;
    for (x = left, xm = 0; x < std::min(left + WIDTH, bg.cols); x++, xm++) {
        for (y = top, ym=0; y < std::min(top + HEIGHT, bg.rows); y++, ym++) {
            auto pixel = mat.at<Vec4b>(cv::Point(xm, ym));
            if (pixel.val[3] > 128) {
                bg.at<Vec4b>(cv::Point(x, y)) = pixel;
            }
        }
    }
}

string serialize(Mat &&subImg, int32_t label) {
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

void addBackground(Json::Value &value, vector<string> & data)  {
    std::set<ImgPoint> points;
    ImageInfo imageInfo(value);

    auto types = value["type"];
    for (auto &type: types) {
        getPoints(*type.begin(),points );
    }

    int left = value["left"].asInt();
    int right = value["right"].asInt();
    int top = value["top"].asInt();
    int bottom = value["bottom"].asInt();

    for (int x=left; x < right; x+=3){
        for (int y=top; y < bottom; y+=3){
            ImgPoint point(x,y);
            if (points.count(point) == 0){
                Mat subImg = imageInfo.subImg(point.x, point.y, WIDTH, HEIGHT);
                data.push_back(serialize(std::move(subImg), 0));
            } else {
                cout << " point (" << x <<"," << y << ") already set" << endl;
            }
        }
    }
}

void getPoints(Json::Value &value, std::set<ImgPoint> & points) {
    std::vector<ImgPoint> result;
    for (auto &point: value) {
        points.insert(ImgPoint(point));
    }
}