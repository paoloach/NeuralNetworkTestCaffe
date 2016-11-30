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

    caffe::Datum next();

private:
    std::vector<TestImages> testImages;
    uint32_t indexVector;
    uint32_t indexImage;
};


#endif //NEURALNETWORKCAFFETEST_TESTIMAGESVECTOR_H
