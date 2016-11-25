//
// Created by paolo on 24/11/16.
//

#ifndef NEURALNETWORKCAFFETEST_LINEPOINT_H
#define NEURALNETWORKCAFFETEST_LINEPOINT_H


#include <algorithm>

class LinePoint {
public:
    LinePoint(int a, int b):min{std::min(a,b)},max{std::max(a,b)}{}
    LinePoint(){}
    int min;
    int max;
};


#endif //NEURALNETWORKCAFFETEST_LINEPOINT_H
