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
#include <caffe/layers/memory_data_layer.hpp>

#include "caffe/proto/caffe.pb.h"
#include "cuda.h"
#include "TestImagesVector.h"

using std::vector;
using std::string;
using namespace caffe;
using boost::dynamic_pointer_cast;


static std::string modelName = "./learn/net-memory.prototxt";
static std::string dataFile = "data_min.txt";
static int iterations = 10000;


// Parse GPU ids or use all available devices
static void get_gpus(vector<int> *gpus) {
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
    vector<string> stages();


    TestImagesVector testImagesVector(dataFile);

    LOG(INFO) << "size: " << testImagesVector.size() << std::endl;
    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    std::cout << "GPU device name: " << device_prop.name;
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    // Instantiate the caffe net.
    Net<float> caffe_net(modelName, TEST);

    auto imagestest = dynamic_pointer_cast<MemoryDataLayer<float>>(caffe_net.layer_by_name("imagestest"));
    if (!imagestest) {
        std::cerr << "Unable to find imagestest layer in " << modelName << std::endl;
        exit(-1);
    }

    vector<Datum> datum_vector;
    datum_vector.push_back(testImagesVector.next());
    imagestest->AddDatumVector(datum_vector);


    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    float iter_loss;
    const vector<Blob<float> *> &result = caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
        const float *result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k, ++idx) {
            const float score = result_vec[k];
            test_score.push_back(score);
            test_score_output_id.push_back(j);
            const std::string &output_name = caffe_net.blob_names()[
                    caffe_net.output_blob_indices()[j]];
            LOG(INFO) << output_name << " = " << score;
        }
    }
    loss /= iterations;
    std::cout << "Loss: " << loss;
    for (int i = 0; i < test_score.size(); ++i) {
        const std::string &output_name = caffe_net.blob_names()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight = caffe_net.blob_loss_weights()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * mean_score << " loss)";
        }
        std::cout << output_name << " = " << mean_score << loss_msg_stream.str();
    }

    return 0;
}

int main(int argc, char **argv) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Google logging.
    ::google::InitGoogleLogging((argv)[0]);
    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();
    std::cout << "version: " << ::google::VersionString() << std::endl;
    test();
}