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


static std::string modelName = "./learn/net.prototxt";
static std::string modelNameTest = "./learn/net-memory.prototxt";
static std::string solverName = "./learn/solver.prototxt";
static std::string snapshot = "_iter_146798.caffemodel";
static std::string dataFile = "data_min.txt";
static int iterations = 10000;


caffe::LayerParameter * findLayer(NetParameter & parameter, const string & name);

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
    LOG(INFO) << "GPU device name: " << device_prop.name;
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    // Instantiate the caffe net.



    caffe::SolverParameter solver_param;
    solver_param.mutable_train_state()->set_level(1);
    solver_param.set_solver_mode(caffe::SolverParameter_SolverMode_GPU);
    solver_param.set_device_id(gpus[0]);
    solver_param.set_base_lr(0.001);
    solver_param.set_weight_decay(0.0005);
    solver_param.set_type("Adam");
    solver_param.set_net(modelNameTest);

    auto solverRegistry = caffe::SolverRegistry<float>::CreateSolver(solver_param);
    shared_ptr<caffe::Solver<float> > solver(solverRegistry);
   // solver->Restore(snapshot.c_str());

    Net<float> netTest(modelNameTest, Phase::TEST);
    //solver->net()->ShareTrainedLayersWith(&netText);
   // netTest.ShareTrainedLayersWith(solver->net().get());
    netTest.CopyTrainedLayersFrom(snapshot);


    auto imagestestLayer = dynamic_pointer_cast<MemoryDataLayer<float>>(netTest.layer_by_name("images"));


    int totInaccuracy = 0;
    for (int i=0; i< testImagesVector.size(); i++) {
        vector<Datum> datum_vector;
        auto testImageElement = testImagesVector.next();
        auto datum = std::get<0>(testImageElement);
        datum_vector.push_back(datum);
        imagestestLayer->AddDatumVector(datum_vector);

        float accuracy;
        float loss;
        const vector<Blob<float> *> &result = netTest.Forward();
        for (int j = 0; j < result.size(); ++j) {
            const float *result_vec = result[j]->cpu_data();
            for (int k = 0; k < result[j]->count(); ++k) {
                const float score = result_vec[k];
                const std::string &output_name = netTest.blob_names()[netTest.output_blob_indices()[j]];
                if (output_name == "accuracy")
                    accuracy = score;
                if (output_name == "loss"){
                    loss = score;
                }
                if (accuracy >=0) {
                    totInaccuracy++;
                 //   cv::imwrite("imagesTmp/backImg" + std::to_string(totInaccuracy) + ".png", std::get<1>(testImageElement));
                    LOG(INFO) << "accuracy: " << accuracy << ", loss: " << loss;
                }
            }
        }
    }
    LOG(INFO) << "innacuracy: " << totInaccuracy << " on " << testImagesVector.size() << " test images";
    return 0;
}


int main(int argc, char **argv) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Google logging.
//    ::google::InitGoogleLogging(argv[0]);
    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();
    LOG(INFO) << "version: " << ::google::VersionString() << std::endl;
    test();
}