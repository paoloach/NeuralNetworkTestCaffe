# Try to find the LMBD libraries and headers
#   LMDB_FOUND - system has LMDB lib
#  LMDB_INCLUDE_DIR - the LMDB include directory
#  LMDB_LIBRARIES - Libraries needed to use LMDB

# FindCWD based on FindGMP by:
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.

# Adapted from FindCWD by:
# Copyright 2013 Conrad Steenberg <conrad.steenberg@gmail.com>
# Aug 31, 2013

find_path(LMDB++_INCLUDE_DIR NAMES  lmdb++.h PATHS "$ENV{LMDB_DIR}/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB++ DEFAULT_MSG LMDB++_INCLUDE_DIR )

if(LMDB++_FOUND)
  message(STATUS "Found lmdb++    (include: ${LMDB++_INCLUDE_DIR})")
  mark_as_advanced(LMDB++_INCLUDE_DIR)
endif()
