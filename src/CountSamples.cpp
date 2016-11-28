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
    std::vector<int> counter;
    for (int i=0; i < 10; i++){
        counter.push_back(0);
    }
    while (true){
        int next = mdb_cursor_get(cursor,&key,&val, MDB_NEXT);
        if (next < 0)
            break;

        auto strDatum = string(static_cast<const char*>(val.mv_data), val.mv_size);
        datum.ParseFromString(strDatum);
        if (datum.label() < 10){
            counter[datum.label()]++;
        } else {
            std::cerr << " Wrong label: " << datum.label() << std::endl;
        }
    }
    for(int i=0; i<10; i++){
        cout << "label: " << i << ", count: " << counter[i] << endl;
    }
}
