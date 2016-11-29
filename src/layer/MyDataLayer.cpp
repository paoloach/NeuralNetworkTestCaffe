//
// Created by paolo on 29/11/16.
//

#include "MyDataLayer.h"

MyDataLayer::MyDataLayer()  {
    set_name("data");
    set_type("MemoryData");
    add_top("data");
    add_top("expected");
    caffe::MemoryDataParameter * MemoryDataParameters =  mutable_memory_data_param();
    MemoryDataParameters->set_batch_size(1);
    MemoryDataParameters->set_channels(1);
    MemoryDataParameters->set_height(1);
    MemoryDataParameters->set_width(1);
}
