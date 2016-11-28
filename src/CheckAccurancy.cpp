//
// Created by paolo on 28/11/16.
//


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/proto/caffe.pb.h"
#include "cuda.h"


using std::vector;
using std::string;
using namespace caffe;


static std::string modelName ="./learn/net.prototxt";
static int iterations=10000;


// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
        int count = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDeviceCount(&count));
#else
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
}

// Test: score a model.
int test() {
    vector<string> stages ();

    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() != 0) {
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, gpus[0]);
        std::cout<< "GPU device name: " << device_prop.name;
#endif
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } else {
        std::cout<< "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }
    // Instantiate the caffe net.
    Net<float> caffe_net(modelName, caffe::TEST);
    //caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    std::cout<< "Running for " << iterations << " iterations.";

    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    for (int i = 0; i < iterations; ++i) {
        float iter_loss;
        const vector<Blob<float>*>& result =
                                          caffe_net.Forward(&iter_loss);
        loss += iter_loss;
        int idx = 0;
        for (int j = 0; j < result.size(); ++j) {
            const float* result_vec = result[j]->cpu_data();
            for (int k = 0; k < result[j]->count(); ++k, ++idx) {
                const float score = result_vec[k];
                if (i == 0) {
                    test_score.push_back(score);
                    test_score_output_id.push_back(j);
                } else {
                    test_score[idx] += score;
                }
                const std::string& output_name = caffe_net.blob_names()[
                        caffe_net.output_blob_indices()[j]];
                std::cout<< "Batch " << i << ", " << output_name << " = " << score;
            }
        }
    }
    loss /= iterations;
    std::cout<< "Loss: " << loss;
    for (int i = 0; i < test_score.size(); ++i) {
        const std::string& output_name = caffe_net.blob_names()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight = caffe_net.blob_loss_weights()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * mean_score << " loss)";
        }
        std::cout<< output_name << " = " << mean_score << loss_msg_stream.str();
    }

    return 0;
}

int main(int argc, char** argv) {
    test();
}