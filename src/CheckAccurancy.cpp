//
// Created by paolo on 28/11/16.
//


#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>

#include "TestImagesVector.h"


// ----------------------------------------------------------------
//  Check the learning parameters (in the dataFile file)

using std::vector;
using std::string;
using namespace caffe;
using boost::dynamic_pointer_cast;
using boost::shared_ptr;



static std::string modelNameTest = "./learn/net-memory.prototxt";
static std::string snapshot = "./learn/adam_iter_1000000.caffemodel";
static std::string dataFile = "data_min.txt";


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
    Net<float> netTest(modelNameTest, Phase::TEST);
    netTest.CopyTrainedLayersFrom(snapshot);

    auto imagestestLayer = dynamic_pointer_cast<MemoryDataLayer<float>>(netTest.layer_by_name("images"));

    LOG(INFO) << "Total images: " << testImagesVector.size();
    int failImages=0;
    for (int i=0; i< testImagesVector.size(); i++) {
        vector<Datum> datum_vector;
        auto testImageElement = testImagesVector.next();
        auto datum = std::get<0>(testImageElement);
        datum_vector.push_back(datum);
        imagestestLayer->AddDatumVector(datum_vector);

        const vector<Blob<float> *> &results = netTest.Forward();
        auto & outputBlobIndices = netTest.output_blob_indices();
        float expectedLabel = -1;
        float found[6] {-1,-1,-1,-1,-1,-1};
        for (int j = 0; j < results.size(); ++j) {
            const float *data = results[j]->cpu_data();
            const std::string &blobName = netTest.blob_names()[outputBlobIndices[j]];

            if (blobName == "label"){
                expectedLabel = data[0];
            } else if (blobName == "ip1"){
                for(int i=0; i <6; i++){
                    found[i] = data[i];
                }
            } else {
                LOG(INFO) << "blob name: " << blobName;
            }
        }
        if (expectedLabel == -1){
            LOG(WARNING) << "label not found";
        } else if (found[0] == -1 && found[1] == -1 && found[2] == -1 && found[3] == -1 && found[4] == -1 && found[5] == -1  ){
            LOG(WARNING) << "lp1 not found";
        } else {

            auto max = std::max_element(std::begin(found),std::end(found) );
            int index = max-found;
            if (index != expectedLabel) {
                failImages++;
                LOG(INFO) << "expected: " << expectedLabel << ", found (" << found[0] << "," << found[1] << ","
                          << found[2] << "," << found[3] << "," << found[4] << "," << found[5] << ")";
            cv::imwrite("imagesTmp/" + std::to_string(failImages) + "_" + std::to_string(index)+".png",std::get<1>(testImageElement) );
            }
        }


    }
    LOG(INFO) << "Total fail images: " << failImages;
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