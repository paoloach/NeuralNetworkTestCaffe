//
// Created by paolo on 21/11/16.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <lmdb.h>

#include "proto/test.pb.h"

using std::cout;
using std::endl;
using std::string;
using namespace cv;


int main(int argc, char **argv){
    if (argc != 3) {
        std::cerr << "need data file and output folder" << endl;
        exit(-1);
    }

    MDB_env *env;
    mdb_env_create(&env);
    mdb_env_open(env, argv[1], 0, 0664);
    MDB_dbi dbi;
    MDB_txn *txn;
    MDB_cursor *cursor;
    MDB_val  key;
    MDB_val  val;

    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_open(txn, NULL, 0, &dbi);
    mdb_cursor_open(txn,dbi,&cursor);
    genSamples::Datum datum;
    int count=0;
    while (true){
        int next = mdb_cursor_get(cursor,&key,&val, MDB_NEXT);
        if (next < 0)
            break;

        auto strDatum = string(static_cast<const char*>(val.mv_data), val.mv_size);
        datum.ParseFromString(strDatum);
        cout << datum.width() << "x" << datum.height() << "x" << datum.channels() << endl;
        string fileName = string(argv[2]) + "/img" + std::to_string(count)+"-" + std::to_string(datum.label())+".png";
        cout << "size: " << datum.data().size() << endl;
        const char * dataIter = datum.data().c_str();
        Size size(datum.width(),datum.height());
        Mat newImg = Mat(size, CV_8UC3, (void *)dataIter);
        imwrite(fileName, newImg) ;
        count++;
    };
    cout << "Written " << count << " images" << endl;
}
