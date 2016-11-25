//
// Created by paolo on 24/11/16.
//


#include <opencv2/opencv.hpp>
#include <map>
#include <tuple>

#include "LinePoint.h"

using namespace cv;
using std::map;
using std::min;
using std::max;

const char *filename = "/home/paolo/workspace/NeuralNetworkCaffeTest/images/01-08-2016.jpg";

map<int, LinePoint> removeDuplicate(map<int, LinePoint>  & map);

std::map<int, LinePoint>::iterator   findLine(std::map<int, LinePoint>  & lines, int min);

int main(int argc, char **argv) {
    Mat dst, color_dst;

    auto src = imread(filename, 0);
    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, color_dst, COLOR_GRAY2BGR);



    std::vector<Vec4i> lines;
    std::map<int, LinePoint> vLines;
    std::map<int, LinePoint> hLines;
    HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
    for(auto & line: lines){
        if (line[0] == line[2]){
            vLines[line[0]] = LinePoint(line[1],line[3]);
        }
        if (line[1] == line[3]){
            hLines[line[1]] = LinePoint(line[0],line[2]);
        }
    }
    for (auto & vLine: vLines){
        std::cout << vLine.first << "(" << vLine.second.min<< "," << vLine.second.max  << ")" << std::endl;
    }

    for (auto & vLine: hLines){
        std::cout << vLine.first << "(" << vLine.second.min<< "," << vLine.second.max  << ")" << std::endl;
    }
//    vLines = removeDuplicate(vLines);
//    hLines = removeDuplicate(hLines);
    for (auto & vLine: vLines){
        auto top = findLine(hLines, vLine.second.min);
        auto bottom = findLine(hLines, vLine.second.max);
        if (top != hLines.end() && bottom != hLines.end()){
            std::cout << "rect: {" <<   vLine.first << " , " << vLine.second.min << " - " <<  top->second.max << " , " << vLine.second.max << std::endl;
        }
    }


//    namedWindow("source", 1);
//    imshow("source", src);
//
//    namedWindow("Detected", 1);
//    imshow("Detected", dst);

    waitKey(0);
}


std::map<int, LinePoint>::iterator  findLine(std::map<int, LinePoint>  & lines, int min){
    for (const auto & line: lines){
        if (abs(line.first - min) < 4){
            return lines.find(line.first);
        }
    }
    return lines.end();
}

map<int, LinePoint> removeDuplicate(std::map<int, LinePoint> & lines) {
    map<int, LinePoint> result;
    int previousKey = lines.begin()->first;
    auto previousTuple = lines.begin()->second;
    result.insert(*lines.begin());
    for (auto & line: lines){
        if (abs(line.first -previousKey) > 4){
            result.insert({previousKey, previousTuple});
            previousKey = line.first;
            previousTuple = line.second;
        } else {
            previousTuple = LinePoint(min(line.second.min,previousTuple.min), max(line.second.max,previousTuple.max));
            previousKey = line.first;
        }

    }
    result.insert({previousKey, previousTuple});
    return result;
}
