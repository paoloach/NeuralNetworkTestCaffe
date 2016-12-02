//
// Created by paolo on 29/11/16.
//

#ifndef NEURALNETWORKCAFFETEST_TESTIMAGESVECTOR_H
#define NEURALNETWORKCAFFETEST_TESTIMAGESVECTOR_H

#include <string>
#include <vector>
#include "TestImages.h"

class TestImagesVector {
public:
    TestImagesVector(std::string &filename);
    size_t  size();

    std::tuple<caffe::Datum, cv::Mat> next();

private:
    std::vector<TestImages> testImages;
    std::map<std::string, int> mapTypes;
    uint32_t indexVector;
    uint32_t indexImage;
};


#endif //NEURALNETWORKCAFFETEST_TESTIMAGESVECTOR_H
