#!/usr/bin/env sh
set -e

TOOLS=/home/paolo/workspace/caffe/cmake-build-debug/tools

$TOOLS/caffe-d train --solver=solver.prototxt $@


