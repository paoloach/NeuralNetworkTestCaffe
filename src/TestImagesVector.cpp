//
// Created by paolo on 29/11/16.
//

#include <fstream>

#include "./json/json/json.h"
#include "TestImagesVector.h"

TestImagesVector::TestImagesVector(std::string &filename) :indexVector(0), indexImage(0){
    std::ifstream iStream(filename);
    Json::Value root;

    iStream >> root;

    for (auto & child: root){
        auto types = child["type"];
        for (auto &type: types) {
            auto label = type.begin().key().asString();
            if (mapTypes.count(label) == 0){
                mapTypes[label] = mapTypes.size()+1;
            }
        }
    }

    for (auto &child: root) {
        testImages.emplace_back(child);
    }

}

size_t TestImagesVector::size() {
    size_t s=0;
    for( auto & testImage: testImages){
        s+=testImage.size();
    }
    return s;
}

std::tuple<caffe::Datum, cv::Mat> TestImagesVector::next() {
    indexImage++;
    if (testImages[indexVector].size() <= indexImage){
        indexImage=0;
        indexVector++;
        if (indexVector >= testImages.size()){
            indexVector=0;
        }
    }
    return testImages[indexVector].get(indexImage);
}
