//
// Created by paolo on 25/11/16.
//

#include <vector>
#include <string>
#include <iostream>
#include <lmdb.h>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using cv::imread;
using cv::Mat;

vector<string> typeImagesPath = {"images/circleGree.png","images/squareYellow.png","images/circleGreen.png"};
vector<string> background = {"images/background1.png",
                                  "images/background2.png",
                                  "images/background3.png",
                                  "images/background4.png",
                                  "images/background5.png",
                                  "images/background6.png",
                                  "images/background7.png",
                                  "images/background8.png",};

Mat addBackround(Mat mat, Mat mat1);

int main(int argc, char **argv) {
    vector<Mat> images;
    vector<Mat> imagesBackground;

    if (argc != 2) {
        std::cerr << "need output directory" << std::endl;
        exit(-1);
    }

    MDB_env *env= nullptr;

    struct MDB_envinfo current_info;

    MDB_dbi dbi;
    MDB_txn *txn= nullptr;
    int counter = 0;

//    mdb_env_create(&env);
//    mdb_env_set_mapsize(env, 100*1024*1024);
//
//    mdb_env_open(env, argv[2], 0, 0664);
//    mdb_txn_begin(env, NULL, 0, &txn);
//    mdb_open(txn, NULL, 0, &dbi);

    for (const auto & image : typeImagesPath){
        images.push_back(imread(image, CV_LOAD_IMAGE_UNCHANGED));
    }
    for (const auto & image : background){
        imagesBackground.push_back(imread(image, CV_LOAD_IMAGE_UNCHANGED));
    }
    auto a = imagesBackground[3].clone();
    auto b = images[0].clone();
    Mat dest =addBackround(images[0].clone(),a);

    cout << "b channel:  " << b.channels() << ", a channel" << a.channels() << endl;
    cv::namedWindow("source", 1);
    cv::imshow("source", a );
    //cv::waitKey(0);
}

Mat addBackround(Mat mat, Mat background) {
    mat.copyTo(background);
    return background;
}