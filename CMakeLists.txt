cmake_minimum_required(VERSION 3.5)

project(GrorReg)

set(CMAKE_BUILD_TYPE "Release")
#set(PCL_DIR "D:/ThirdLib/PCL 1.8.1/cmake")
#set(PCL_DIR "C:/Program Files/PCL 1.12.0/cmake")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS} -D_EXPORTING; -D_CRT_SECURE_NO_WARNINGS;  -DPCL_NO_PRECOMPILE)
link_directories(${PCL_LIBRARIES})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FILE (GLOB ALL_SOURCES "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/src/*.cpp" )
include_directories( ${PROJECT_SOURCE_DIR}/include)
# exe
add_executable(GrorReg ${ALL_SOURCES})
target_link_libraries(GrorReg  ${PCL_LIBRARIES})
