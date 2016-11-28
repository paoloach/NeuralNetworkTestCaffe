cmake_minimum_required(VERSION 3.6)
project(NeuralNetworkCaffeTest)


set (CAFFE_DIR_INCLUDE /home/paolo/workspace/caffe/include)
set (CAFFE_DIR_LIBRARY /home/paolo/workspace/caffe/.build_release/lib)

set (CUDA_INCLUDE_DIR /opt/cuda/include)
set (CUDA_LIB_DIR /opt/cuda/lib)

include(cmake/Utils.cmake)
include(cmake/ProtoBuf.cmake)
#include(cmake/LMDBConfig.cmake)
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( '.')

set(LMDB_DIR ./cmake)

find_package(LMDB REQUIRED)
include_directories(SYSTEM ${LMDB_INCLUDE_DIR}  ${CAFFE_DIR_INCLUDE} ${CUDA_INCLUDE_DIR})
link_directories(${CAFFE_DIR_LIBRARY} ${CUDA_LIB_DIR})



set(proto_files
        proto/test.proto)
set(proto_gen_folder ${CMAKE_CURRENT_SOURCE_DIR}/src/proto)

caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        ${proto_srcs}
        ${proto_hdrs}
        src/json/jsoncpp.cpp
        src/json/json/json.h
        src/json/json/json-forwards.h
        main.cpp)

set(COUNT_SAMPLES_FILES
        ${proto_srcs}
        ${proto_hdrs}
        src/CountSamples.cpp)

set(DUMP_IMPAGES_FILES
        ${proto_srcs}
        ${proto_hdrs}
        src/DumpImages.cpp src/MakeSamples.cpp)


set(MAKE_SAPLES
        ${proto_srcs}
        ${proto_hdrs}
        src/json/jsoncpp.cpp
        src/json/json/json.h
        src/json/json/json-forwards.h
        src/MakeSamples.cpp
        )

set(MAKE_CHECK_ACCURANCY
        ${proto_srcs}
        ${proto_hdrs}
        src/CheckAccurancy.cpp)

set (TEST_FIND src/TestFindRect.cpp src/proto/LinePoint.h)
add_executable(NeuralNetworkCaffeTest ${SOURCE_FILES})
add_executable(DumpImages ${DUMP_IMPAGES_FILES})
add_executable(CountSamples ${COUNT_SAMPLES_FILES})
#add_executable(TestFindRect ${TEST_FIND})
add_executable(MakeSamples ${MAKE_SAPLES} )
add_executable(CheckAccurancy ${MAKE_CHECK_ACCURANCY} )

target_link_libraries(NeuralNetworkCaffeTest
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )

target_link_libraries(DumpImages
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )

target_link_libraries(CountSamples
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )

target_link_libraries(CheckAccurancy
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )


#target_link_libraries(TestFindRect
#        ${OpenCV_LIBS}
#        )

target_link_libraries(MakeSamples
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )