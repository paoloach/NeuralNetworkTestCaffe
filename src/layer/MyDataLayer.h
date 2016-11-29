//
// Created by paolo on 29/11/16.
//

#ifndef NEURALNETWORKCAFFETEST_MYDATALAYER_H
#define NEURALNETWORKCAFFETEST_MYDATALAYER_H

#include<caffe/caffe.hpp>
#include<caffe/layers/data_layer.hpp>

class MyDataLayer : public caffe::LayerParameter{
public:
    MyDataLayer();
};


#endif //NEURALNETWORKCAFFETEST_MYDATALAYER_H
