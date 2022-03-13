#pragma once
//windows
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/filter.h>
#include <pcl/features/3dsc.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/flann_search.h>
#include <pcl/search/pcl_search.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>