//
// Created by paolo on 25/11/16.
//

#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <iostream>
#include <lmdb.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "./proto/test.pb.h"
#include "./json/json/json.h"
#include <H5Cpp.h>
#include <H5LTpublic.h>

constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;

using std::string;
using std::vector;
using std::tuple;
using std::cout;
using std::endl;
using namespace cv;
using namespace H5;

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


const char *datasetName = "data";
const char *labelsetName = "label";

vector<string> typeImagesPath = {"images/circleGreen.png", "images/squareYellow.png",
                                 "images/squarePurple.png", "images/circleBlue.png", "images/triangleYellow.png"};
vector<Scalar> lineColors = {{0x41, 0x5c, 0x4c, 0xFF},
                             {0x75, 0x4c, 0x2b, 0xFF},
                             {0x3c, 0x61, 0x4d, 0xFF},
                             {0xdb, 0x9a, 0x73, 0xFF},
                             {0x73, 0x44, 0x28, 0xFF}};
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

std::random_device rd;
std::default_random_engine randomEngine(rd());
std::uniform_int_distribution<int> uniform_dist_0_7(0, 7);
std::uniform_int_distribution<int> uniform_dist_0_WIDTH(0, WIDTH);
std::uniform_int_distribution<int> uniform_dist_0_channel(0, 2);
std::uniform_int_distribution<int> uniform_dist_0_delta(-10, 10);
std::uniform_int_distribution<int> uniform_dist_0_HEIGHT(0, HEIGHT);
std::uniform_real_distribution<float> uniform_dist_0_180(0, (float) M_PI);

//vector<string> typeImagesPath = {"images/circleGreen.png", "images/squareYellow.png",};
//vector<string> background = {"images/background1.png"};



Mat addBackround(Mat &mat, Mat &mat1);

void add(Mat &bg, Mat &mat, int left, int top);

void addLine(Mat &bg, Scalar colorLine, int left, int top);

void addNoise(Mat &img);

std::tuple<vector<Mat>, vector<Mat>>
createImage(Mat &imageBG, Scalar &colorLine, Mat &imageType, Mat &imageOther, int xOther, int yOther);

void save(const string &dataToWrite, MDB_txn *txn, MDB_dbi &dbi, int &counter);

void addBackground(Json::Value &value, vector<tuple<Mat, int >> &data);

void getPoints(Json::Value &value, std::set<ImgPoint> &points);

int main(int argc, char **argv) {
    vector<Mat> images;
    vector<Mat> imagesBackground;
    vector<tuple<Mat, int >> data;


    if (argc != 3) {
        std::cerr << "need output directory" << std::endl;
        exit(-1);
    }


    std::ifstream iStream(argv[2]);
    Json::Value root;

    iStream >> root;

    int counter = 0;

    for (const auto &image : typeImagesPath) {
        auto mat = imread(image, CV_LOAD_IMAGE_UNCHANGED);
        images.push_back(mat);
    }
    for (const auto &image : background) {
        imagesBackground.push_back(imread(image, CV_LOAD_IMAGE_UNCHANGED));
    }

    int typeCounter = 1;
    for (int indexImage = 0; indexImage < images.size(); indexImage++) {
        auto &image = images[indexImage];
        auto &colorLine = lineColors[indexImage];
        for (auto &other: images) {
            if (&image != &other) {
                for (auto &imageBG: imagesBackground) {
                    data.emplace_back(addBackround(image, imageBG), typeCounter);
                    for (int x = -WIDTH / 2 + 4; x < WIDTH + WIDTH / 2 - 4; x++) {
                        for (int y = -HEIGHT / 2 + 4; y < HEIGHT + HEIGHT / 2 - 4; y++) {
                            if (abs(x - WIDTH / 2) < 2 && abs(y - HEIGHT / 2) < 2)
                                continue;
                            counter++;
                            auto newImage = createImage(imageBG, colorLine, image, other, x, y);
                            for (auto &image: std::get<0>(newImage)) {
                                data.emplace_back(image, typeCounter);
                            }
                            for (auto &bg: std::get<1>(newImage)) {
                                data.emplace_back(bg, 0);
                            }
                            counter++;
                        }
                    }
                }
            }
        }
        typeCounter++;
    }

//    cout << "Save tmp Data" << endl;
//    int c=0;
//    for (auto & image: data){
//        auto type = std::get<1>(image);
//        auto & imageDAta = std::get<0>(image);
//        if (type ==1 || type==0){
//            imwrite("imagesTmp/im" + std::to_string(c)+"_"+std::to_string(type)+".png", imageDAta);
//            c++;
//        }
//        if (c > 100)
//            exit(0);
//    }
//    cout << "End save tmp Data" << endl;


    for (auto &child: root) {
        if (!child["test"].asInt()) {
            cout << "data image from " << child["file"] << endl;
            addBackground(child, data);
        }
    }

    cout << "created " << data.size() << " samples" << endl;
    cout << "now shuffling" << endl;
    std::random_shuffle(std::begin(data), std::end(data));
    string testFile = string(argv[1]) + ".test";
    cout << "now saving" << endl;


    uint64 chunkSize = 100000;
    counter = 0;
    std::stringstream sstream;
    for (int i = 0; i < data.size(); i += chunkSize) {
        string fileName = "data_" + std::to_string(i) + ".hdf5";
        sstream << "../" << fileName << "\n";
        std::cout << "writing "<< fileName << std::endl;
        uint64_t dataSize = std::min(std::abs(data.size() - i), chunkSize);
        size_t dataMemory = dataSize * 3 * WIDTH * HEIGHT;
        size_t labelMemory = dataSize;
        float *dataH5F = new float[dataMemory];
        float *dataIter = dataH5F;
        float *labelH5F = new float[labelMemory];
        float *labelIter = labelH5F;
        for(auto iterData = data.begin()+i; iterData != data.begin()+dataSize+i; iterData++){
            float *start = dataIter;
            Mat matrix = std::get<0>(*iterData);
            int type = std::get<1>(*iterData);
            auto iter = matrix.begin<Vec4b>();
            while (iter != matrix.end<Vec4b>()) {
                auto pixel = (*iter);
                *dataIter = pixel.val[0];
                dataIter++;
                *dataIter = pixel.val[1];
                dataIter++;
                *dataIter = pixel.val[2];
                dataIter++;
                iter += 1;
            }

            *labelIter = type;
            labelIter++;
        }

        auto h5file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        hsize_t dimsData[4] = {dataSize, 3, WIDTH, HEIGHT};
        hsize_t dimsLabel[2] = {dataSize, 1};
        H5LTmake_dataset_float(h5file, datasetName, 4, dimsData, dataH5F);
        H5LTmake_dataset_float(h5file, labelsetName, 2, dimsLabel, labelH5F);
        H5Fclose(h5file);
        std::cout << "Written " << fileName << std::endl;
    }

    std::ofstream hdf5files("hd5files2.txt");
    if (!hdf5files){
        std::cerr << "Error open learn/hd5files2.txt" << std::endl;
    }
    hdf5files << sstream.str();
    if (!hdf5files){
        std::cerr << "Error writing learn/hd5files2.txt" << std::endl;
    }
    hdf5files.close();


    cout << "finished" << endl;
}

std::tuple<vector<Mat>, vector<Mat>>
createImage(Mat &imageBG, Scalar &colorLine, Mat &imageType, Mat &imageOther, int xOther, int yOther) {
    Mat bg = imageBG.clone();
    vector<Mat> background;
    vector<Mat> images;
    addLine(bg, colorLine, WIDTH / 2, HEIGHT / 2);
    add(bg, imageType, WIDTH / 2, HEIGHT / 2);
    add(bg, imageOther, xOther, yOther);
    Mat imageTmp = bg(cv::Rect(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT));
    for (int i = 0; i < 10; i++) {
        Mat image = imageTmp.clone();
        addNoise(image);
        images.push_back(image);
    }
    for (int x = std::max(xOther, WIDTH / 2); x > std::min(xOther, WIDTH / 2); x-=2) {
        for (int y = std::max(yOther, HEIGHT / 2); y > std::min(yOther, HEIGHT / 2); y-=2) {
            if (x >= 0 && y >= 0 && x < WIDTH && y < HEIGHT) {
                Mat bgImg = bg(cv::Rect(x, y, WIDTH, HEIGHT)).clone();
                addNoise(bgImg);
                background.push_back(bgImg);
            }
        }
    }
    return std::make_tuple(images, background);
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

void addNoise(Mat &img) {
    for (int i = 0; i < 30; i++) {
        Point point(uniform_dist_0_WIDTH(randomEngine), uniform_dist_0_HEIGHT(randomEngine));
        int channel = uniform_dist_0_channel(randomEngine);
        int delta = uniform_dist_0_delta(randomEngine);
        if (point.x >=0 && point.x < img.cols && point.y >= 0 && point.y < img.rows) {
            auto &color = img.at<Vec4b>(point);
            if (color[channel] + delta < 0) {
                color[channel] = 0;
            } else if (color[channel] + delta > 255) {
                color[channel] = 255;
            } else {
                color[channel] += delta;
            }
        }
    }
}

void add(Mat &bg, Mat &mat, int left, int top) {
    int x, y;
    int xm, ym;
    for (x = left, xm = 0; x < std::min(left + WIDTH, bg.cols); x++, xm++) {
        for (y = top, ym = 0; y < std::min(top + HEIGHT, bg.rows); y++, ym++) {
            auto pixel = mat.at<Vec4b>(cv::Point(xm, ym));
            if (pixel.val[3] > 128) {
                if (x>=0 && x < bg.cols && y >= 0 && y < bg.rows)
                    bg.at<Vec4b>(cv::Point(x, y)) = pixel;
            }
        }
    }
}

void addLine(Mat &bg, Scalar colorLine, int left, int top) {
    Point center((left + WIDTH / 2), (top + HEIGHT / 2));
    int type = uniform_dist_0_7(randomEngine);
    if (type > 0) {
        float angle = uniform_dist_0_180(randomEngine);
        int x = center.x + 2 * WIDTH * sin(angle);
        int y = center.y + 2 * HEIGHT * cos(angle);
        line(bg, center, Point(x, y), colorLine, 1, LINE_AA);
    }
    if (type < 7) {
        float angle = uniform_dist_0_180(randomEngine);
        int x = center.x - 2 * WIDTH * sin(angle);
        int y = center.y + 2 * HEIGHT * cos(angle);
        line(bg, center, Point(x, y), colorLine, 1, LINE_AA);
    }
}

void addBackground(Json::Value &value, vector<tuple<Mat, int >> &data) {
    std::set<ImgPoint> points;
    ImageInfo imageInfo(value);

    auto types = value["type"];
    for (auto &type: types) {
        getPoints(*type.begin(), points);
    }

    int left = value["left"].asInt();
    int right = value["right"].asInt();
    int top = value["top"].asInt();
    int bottom = value["bottom"].asInt();

    for (int x = left; x < right; x += 3) {
        for (int y = top; y < bottom; y += 3) {
            ImgPoint point(x, y);
            if (points.count(point) == 0) {
                Mat subImg = imageInfo.subImg(point.x, point.y, WIDTH, HEIGHT);
                data.emplace_back(subImg, 0);
            } else {
                cout << " point (" << x << "," << y << ") already set" << endl;
            }
        }
    }
}

void getPoints(Json::Value &value, std::set<ImgPoint> &points) {
    std::vector<ImgPoint> result;
    for (auto &point: value) {
        points.insert(ImgPoint(point));
    }
}