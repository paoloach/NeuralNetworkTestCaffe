//
// Created by paolo on 16/12/16.
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
#include <opencv2/opencv.hpp>
#include <map>
#include <tuple>

#include "LinePoint.h"

DEFINE_string(image, "", "The image to analyze.");
DEFINE_string(model, "", "The model file.");
DEFINE_string(learn, "", "The learn data file.");


using namespace cv;
using namespace boost;
using namespace caffe;
using std::map;
using std::vector;
using std::string;

const int WIDTH=16;
const int HEIGHT=16;
const int LABELS=6;

static void analyze();
static caffe::Datum createDatum(Mat &&subImg);
static void get_gpus(vector<int> *gpus);
static Rect findImageData(Mat & image);
static map<int, LinePoint>::iterator  findLine(map<int, LinePoint>  & lines, int min);
static map<int, vector<Point>>  test(Mat image);
static int getMaxIndex(const float * data);


int main(int argc, char **argv) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Google logging.
//    ::google::InitGoogleLogging(argv[0]);
    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();
    LOG(INFO) << "version: " << ::google::VersionString() << std::endl;
    analyze();
}


void analyze(){
    LOG(INFO) << "Analyzing '" << FLAGS_image << "'";
    Mat image = imread(FLAGS_image);
    if (image.empty()){
        LOG(ERROR) << "Unable to load image '" << FLAGS_image << "'";
        exit(-1);
    }

    Rect dataArea = findImageData(image);
    Mat imageData = image(dataArea);
    auto result = test(imageData);
    for(auto & type: result){
        LOG(INFO) << "label: " << type.first;
        for (auto & pos: type.second){
            LOG(INFO) << pos;
        }
    }
}

static Rect findImageData(Mat & image) {
    Mat dst, color_dst;
    Canny(image, dst, 50, 200, 3);
    cvtColor(dst, color_dst, COLOR_GRAY2BGR);

    vector<Vec4i> lines;
    map<int, LinePoint> vLines;
    map<int, LinePoint> hLines;
    HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
    for(auto & line: lines){
        if (line[0] == line[2]){
            vLines[line[0]] = LinePoint(line[1],line[3]);
        }
        if (line[1] == line[3]){
            hLines[line[1]] = LinePoint(line[0],line[2]);
        }
    }
//    vLines = removeDuplicate(vLines);
//    hLines = removeDuplicate(hLines);
    Rect max;
    for (auto & vLine: vLines){
        auto top = findLine(hLines, vLine.second.min);
        auto bottom = findLine(hLines, vLine.second.max);
        if (top != hLines.end() && bottom != hLines.end()){
            int l =  vLine.first;
            int t = vLine.second.min;
            int r = top->second.max;
            int b = vLine.second.max;

            Rect rect(l, t, (r-l), (b-t));

            LOG(INFO) << "rect: " << rect;
            if (rect.area() > max.area()){
                max = rect;
            }
        }
    }
    LOG(INFO) << "Max rect: " << max;
    return max;
}


std::map<int, LinePoint>::iterator  findLine(std::map<int, LinePoint>  & lines, int min){
    for (const auto & line: lines){
        if (abs(line.first - min) < 4){
            return lines.find(line.first);
        }
    }
    return lines.end();
}



// Test: score a model.
map<int, vector<Point>>  test(Mat image) {
    vector<string> stages();
    map<int, vector<Point>> result;

    vector<int> gpus;
    get_gpus(&gpus);
    cudaDeviceProp device_prop;
    if (gpus.size() > 0) {
        cudaGetDeviceProperties(&device_prop, gpus[0]);
        LOG(INFO) << "GPU device name: " << device_prop.name;
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }
    // Instantiate the caffe net.
    Net<float> netTest(FLAGS_model, Phase::TEST);
    netTest.CopyTrainedLayersFrom(FLAGS_learn);

    auto imagestestLayer = dynamic_pointer_cast<MemoryDataLayer<float>>(netTest.layer_by_name("images"));

    int failImages=0;
    int width = image.size[1];
    int height = image.size[0];
    imagestestLayer->set_batch_size(height-HEIGHT);
    for (int x=0; x< width-WIDTH; x++) {
        vector<Datum> datum_vector;
        for (int y=0; y < height-HEIGHT; y++){
            datum_vector.push_back(createDatum(image( Rect(x,y,WIDTH, HEIGHT))));
        }
        imagestestLayer->AddDatumVector(datum_vector);

        const vector<Blob<float> *> &results = netTest.Forward();
        auto & outputBlobIndices = netTest.output_blob_indices();
        for (int j = 0; j < results.size(); ++j) {
            const float *data = results[j]->cpu_data();
            const std::string &blobName = netTest.blob_names()[outputBlobIndices[j]];

            if (blobName == "ip1"){
                for (int index=0; index < results[j]->count(); index+=LABELS){

                    int element = getMaxIndex(data+index);
                    if (element > 0){
                        result[element].push_back(Point(x,index/LABELS));
                    }
                }
            }
        }

    }
    return result;
}


// Parse GPU ids or use all available devices
static void get_gpus(vector<int> *gpus) {
    int count = 0;
    cudaGetDeviceCount(&count);
    LOG(INFO) << "count GPU="<< count;
    for (int i = 0; i < count; ++i) {
        gpus->push_back(i);
    }
}

caffe::Datum createDatum(Mat &&subImg) {
    caffe::Datum datum;
    datum.set_channels(3);
    datum.set_height(HEIGHT);
    datum.set_width(WIDTH);

    std::unique_ptr<uint8_t> data{new uint8_t[WIDTH * HEIGHT * 3]};
    uint8_t *dataIter;
    dataIter = data.get();
    for (auto iter = subImg.begin<Vec3b>(); iter != subImg.end<Vec3b>(); iter++) {
        *dataIter = (*iter).val[0];
        dataIter++;
        *dataIter = (*iter).val[1];
        dataIter++;
        *dataIter = (*iter).val[2];
        dataIter++;
    }
    datum.set_data(data.get(), WIDTH * HEIGHT * 3);
    return datum;
}

static int getMaxIndex(const float * data) {
    int index=0;
    float max=INT_MIN;

    for(int i=0; i < LABELS; i++){
        if (data[i] > max) {
            index = i;
            max = data[index];
        }

    }
    return index;
}