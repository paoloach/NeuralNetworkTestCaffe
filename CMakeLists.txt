cmake_minimum_required(VERSION 3.6)
project(NeuralNetworkCaffeTest)

include(cmake/Utils.cmake)
include(cmake/ProtoBuf.cmake)
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( '.')

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
add_executable(NeuralNetworkCaffeTest ${SOURCE_FILES})

target_link_libraries(NeuralNetworkCaffeTest ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS})