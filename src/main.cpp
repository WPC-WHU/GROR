/**=====================================================================================================
* Copyright 2020, SCHOOL OF GEODESY AND GEOMATIC, WUHAN UNIVERSITY
* WUHAN, CHINA
* All Rights Reserved
* Authors: Pengcheng Wei, Jicheng Dai, et al.
* Do not hesitate to contact the authors if you have any question or find any bugs
* Email: wei.pc@whu.edu.cn
* See LICENSE for the license information
//=======================================================================================================
*/

//windows
#include <iostream>
#include <thread>
#include <chrono>
#include <complex>
//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/distances.h>



#include "ia_gror.h"
#include "gror_pre.h"

int main(int argc, char** argv) {

	//INPUT:
	// 1. path to the source point cloud
	std::string fnameS = argv[1];
	// 2. path to the target point cloud
	std::string fnameT = argv[2];
	// 3. resolution threshold (default 0.1)
	double resolution = atof(argv[3]);
	// 4. optimal threshold (default 800)
	int n_optimal = atoi(argv[4]);
	pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT(new pcl::PointCloud<pcl::PointXYZ>);

	//support pcd or ply
	if (fnameS.substr(fnameS.find_last_of('.') + 1) == "pcd") {
		pcl::io::loadPCDFile(fnameS, *origin_cloudS);
		pcl::io::loadPCDFile(fnameT, *origin_cloudT);
	}
	else if (fnameS.substr(fnameS.find_last_of('.') + 1) == "ply") {
		pcl::io::loadPLYFile(fnameS, *origin_cloudS);
		pcl::io::loadPLYFile(fnameT, *origin_cloudT);
	}

	/*======================================================================*/
	auto t = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::CorrespondencesPtr corr(new pcl::Correspondences);
	Eigen::Vector3f centerS(0, 0, 0), centerT(0, 0, 0);

	GrorPre::grorPreparation(origin_cloudS, origin_cloudT, cloudS, cloudT, issS, issT, corr, resolution);

	auto ShowVGFPointCloud = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 0, 100, 160);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 0, 200, 100);

		//show point clouds after VGF
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerVGF(new pcl::visualization::PCLVisualizer("After Voxel Grid Downsampling"));
		viewerVGF->setBackgroundColor(255, 255, 255);
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
		viewerVGF->spin();
	};

	std::thread vis_thread([cloudS, cloudT, ShowVGFPointCloud]() {ShowVGFPointCloud(cloudS, cloudT); });

	auto t4 = std::chrono::system_clock::now();
	pcl::registration::GRORInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, float> gror;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcs(new pcl::PointCloud<pcl::PointXYZ>);
	gror.setInputSource(issS);
	gror.setInputTarget(issT);
	gror.setResolution(resolution);
	gror.setOptimalSelectionNumber(n_optimal);
	gror.setNumberOfThreads(1);
	gror.setInputCorrespondences(corr);
	gror.align(*pcs);
	auto t5 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of gror: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()) / 1000.0 << std::endl;
	std::cout << "best count: " << gror.getBestCount() << std::endl;
	std::cout << "best final TM: \n" << gror.getFinalTransformation() << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	auto t_end = std::chrono::system_clock::now();
	std::cout << "/*total registration time cost:" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t).count()) / 1000.0 << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	pcl::CorrespondencesPtr est_inliers;
	pcl::CorrespondencesPtr recall_inliers(new pcl::Correspondences);
	pcl::CorrespondencesPtr recall_outliers(new pcl::Correspondences);

	pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloud_S(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloudS, *reg_cloud_S, gror.getFinalTransformation());
	bool is_icp = 0;
	if (is_icp)
	{
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(cloudS);
		icp.setInputTarget(cloudT);
		//icp.setMaxCorrespondenceDistance(0.1);
		//icp.setTransformationEpsilon(1e-5);
		//icp.setEuclideanFitnessEpsilon(0.1);
		//icp.setMaximumIterations(100);
		icp.setUseReciprocalCorrespondences(true);
		icp.align(*reg_cloud_S, gror.getFinalTransformation());
		std::cout << "transformation matrix after ICP: \n" << icp.getFinalTransformation() << std::endl;
		auto t_end = std::chrono::system_clock::now();
		std::cout << "/*total registration time -with icp cost:" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t).count()) / 1000.0 << std::endl;
		std::cout << "/*=================================================*/" << std::endl;
	}


	auto ShowVGFPointCloud2 = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 0, 100, 160);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 255, 85, 0);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerVGF(new pcl::visualization::PCLVisualizer("After Registration"));
		viewerVGF->setBackgroundColor(255, 255, 255);
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
		while (!viewerVGF->wasStopped())
		{
			viewerVGF->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	};

	ShowVGFPointCloud2(reg_cloud_S, cloudT);


}