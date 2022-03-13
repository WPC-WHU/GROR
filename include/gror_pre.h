/**=====================================================================================================
* Copyright 2020, SCHOOL OF GEODESY AND GEOMATIC, WUHAN UNIVERSITY
* WUHAN, CHINA
* All Rights Reserved
* Authors: Pengcheng Wei, Jicheng Dai, et al.
* Do not hesitate to contact the authors if you have any question or find any bugs
* Email: wei.pc@whu.edu.cn
* See LICENSE for the license information
//=======================================================================================================
* Thanks to the work of Cai, et al:
* https://github.com/ZhipengCai/Demo---Practical-optimal-registration-of-terrestrial-LiDAR-scan-pairs
*/

#pragma once
#ifndef GRORPRE_H_
#define GRORPRE_H_

//windows
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>

//PCL
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/pcl_search.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>


namespace GrorPre {
	//voxel grid filter
	void voxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG,
		double inlTh);
	//ISS keypoint extraction
	void issKeyPointExtration(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr iss,
		pcl::PointIndicesPtr iss_Idx, double resolution);
	//FPFH computation
	void fpfhComputation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double resolution, pcl::PointIndicesPtr iss_Idx, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_out);
	//correspondence computation
	void correspondenceSearching(pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfht,
		pcl::Correspondences &corr,
		int max_corr, std::vector<int> &corr_NOs, std::vector<int> &corr_NOt);
	void grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::CorrespondencesPtr corr, double resolution);

	void grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, Eigen::Vector3f &centerS, Eigen::Vector3f &centerT, pcl::CorrespondencesPtr corr, double resolution);
	void centroidTransMatCompute(Eigen::Matrix4f &T, const Eigen::Vector3f &vS, const Eigen::Vector3f &vT);
}
#endif