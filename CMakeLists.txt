cmake_minimum_required(VERSION 3.6)
project(NeuralNetworkCaffeTest)

include(ExternalProject)


set (CAFFE_DIR_INCLUDE /home/paolo/workspace/caffe/include /home/paolo/workspace/caffe/cmake-build-debug/include/)
set (CAFFE_DIR_LIBRARY /home/paolo/workspace/caffe/.build_release/lib)

set (CUDA_INCLUDE_DIR /opt/cuda/include)
set (CUDA_LIB_DIR /opt/cuda/lib)
set (CURA_LIB )

include(cmake/Utils.cmake)
include(cmake/ProtoBuf.cmake)
#include(cmake/LMDBConfig.cmake)
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( '.')

include("cmake/FindGlog.cmake")
include("cmake/FindGFlags.cmake")
include("cmake/glog.cmake")

find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)

set(LMDB_DIR ./cmake)

find_package(CUDA 5.5 QUIET)
find_package(LMDB REQUIRED)
include_directories(SYSTEM ${LMDB_INCLUDE_DIR}  ${CAFFE_DIR_INCLUDE} ${CUDA_INCLUDE_DIR} ${GLOG_INCLUDE_DIRS}  ${Boost_INCLUDE_DIR})
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

set(LAYERS_SRC src/layer/MyDataLayer.cpp  )
set(LAYERS_HEADERS src/layer/MyDataLayer.h)

set (TEST_FIND src/TestFindRect.cpp src/proto/LinePoint.h)
add_executable(NeuralNetworkCaffeTest ${SOURCE_FILES})
add_executable(DumpImages ${DUMP_IMPAGES_FILES})
add_executable(CountSamples ${COUNT_SAMPLES_FILES})
#add_executable(TestFindRect ${TEST_FIND})
add_executable(MakeSamples ${MAKE_SAPLES} )
add_executable(CheckAccurancy ${MAKE_CHECK_ACCURANCY} ${LAYERS_SRC} )

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
        caffe
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        ${GLOG_LIBRARIES}
        ${Boost_LIBRARIES}
        )


#target_link_libraries(TestFindRect
#        ${OpenCV_LIBS}
#        )

target_link_libraries(MakeSamples
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        ${LMDB_LIBRARIES}
        )