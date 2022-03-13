#include "gror_pre.h"

void GrorPre::voxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG, double inlTh)
{
	//format for filtering
	pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2());
	pcl::PCLPointCloud2::Ptr cloudVG2(new pcl::PCLPointCloud2());
	pcl::toPCLPointCloud2(*cloud, *cloud2);
	//set up filtering parameters
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud2);
	sor.setLeafSize(inlTh, inlTh, inlTh);
	//filtering process
	sor.filter(*cloudVG2);
	pcl::fromPCLPointCloud2(*cloudVG2, *cloudVG);

}

void GrorPre::issKeyPointExtration(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ISS, pcl::PointIndicesPtr ISS_Idx, double resolution)
{
	double iss_salient_radius_ = 6 * resolution;
	double iss_non_max_radius_ = 4 * resolution;
	//double iss_non_max_radius_ = 2 * resolution;//for office
	//double iss_non_max_radius_ = 9 * resolution;//for railway
	double iss_gamma_21_(0.975);
	double iss_gamma_32_(0.975);
	double iss_min_neighbors_(4);
	int iss_threads_(1); //switch to the number of threads in your cpu for acceleration

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

	iss_detector.setSearchMethod(tree);
	iss_detector.setSalientRadius(iss_salient_radius_);
	iss_detector.setNonMaxRadius(iss_non_max_radius_);
	iss_detector.setThreshold21(iss_gamma_21_);
	iss_detector.setThreshold32(iss_gamma_32_);
	iss_detector.setMinNeighbors(iss_min_neighbors_);
	iss_detector.setNumberOfThreads(iss_threads_);
	iss_detector.setInputCloud(cloud);
	iss_detector.compute(*ISS);
	ISS_Idx->indices = iss_detector.getKeypointsIndices()->indices;
	ISS_Idx->header = iss_detector.getKeypointsIndices()->header;

}

void GrorPre::fpfhComputation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double resolution, pcl::PointIndicesPtr iss_Idx, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_out)
{
	//compute normal
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(3 * resolution);
	ne.compute(*normal);

	//compute fpfh using normals
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
	fpfh_est.setInputCloud(cloud);
	fpfh_est.setInputNormals(normal);
	fpfh_est.setSearchMethod(tree);
	fpfh_est.setRadiusSearch(8 * resolution);
	fpfh_est.setNumberOfThreads(16);
	fpfh_est.setIndices(iss_Idx);
	fpfh_est.compute(*fpfh_out);
}

void GrorPre::correspondenceSearching(pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfht, pcl::Correspondences & corr, int max_corr, std::vector<int>& corr_NOs, std::vector<int>& corr_NOt)
{
	int n = std::min(max_corr, (int)fpfht->size()); //maximum number of correspondences to find for each source point
	corr.clear();
	corr_NOs.assign(fpfhs->size(), 0);
	corr_NOt.assign(fpfht->size(), 0);
	// Use a KdTree to search for the nearest matches in feature space
	pcl::KdTreeFLANN<pcl::FPFHSignature33> treeS;
	treeS.setInputCloud(fpfhs);
	pcl::KdTreeFLANN<pcl::FPFHSignature33> treeT;
	treeT.setInputCloud(fpfht);
	for (size_t i = 0; i < fpfhs->size(); i++) {
		std::vector<int> corrIdxTmp(n);
		std::vector<float> corrDisTmp(n);
		//find the best n matches in target fpfh
		treeT.nearestKSearch(*fpfhs, i, n, corrIdxTmp, corrDisTmp);
		for (size_t j = 0; j < corrIdxTmp.size(); j++) {
			bool removeFlag = true;
			int searchIdx = corrIdxTmp[j];
			std::vector<int> corrIdxTmpT(n);
			std::vector<float> corrDisTmpT(n);
			treeS.nearestKSearch(*fpfht, searchIdx, n, corrIdxTmpT, corrDisTmpT);
			for (size_t k = 0; k < n; k++) {
				if (corrIdxTmpT.data()[k] == i) {
					removeFlag = false;
					break;
				}
			}
			if (removeFlag == false) {
				pcl::Correspondence corrTabTmp;
				corrTabTmp.index_query = i;
				corrTabTmp.index_match = corrIdxTmp[j];
				corrTabTmp.distance = corrDisTmp[j];
				corr.push_back(corrTabTmp);
				corr_NOs[i]++;
				corr_NOt[corrIdxTmp[j]]++;
			}
		}
	}
}

void GrorPre::grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT,pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::CorrespondencesPtr corr,double resolution)
{
	int max_corr = 5;// neighbor number in descriptor searching
	auto t = std::chrono::system_clock::now();
	/*=============down sample point cloud by voxel grid filter=================*/
	std::cout << "/*voxel grid sampling......" << resolution << std::endl;
	GrorPre::voxelGridFilter(origin_cloudS, cloudS, resolution);
	GrorPre::voxelGridFilter(origin_cloudT, cloudT, resolution);
	
	auto t1 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of cloud down sample : " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	/*=========================extract iss key points===========================*/
	std::cout << "/*extracting ISS keypoints......" << std::endl;
	pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);
	pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);
	GrorPre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);
	GrorPre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);
	auto t2 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of iss key point extraction: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
	std::cout << "/*=================================================*/" << std::endl;


	/*======================fpfh descriptor computation=========================*/
	std::cout << "/*fpfh descriptor computation......" << std::endl;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
	GrorPre::fpfhComputation(cloudS, resolution, iss_IdxS, fpfhS);
	GrorPre::fpfhComputation(cloudT, resolution, iss_IdxT, fpfhT);
	auto t3 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of fpfh descriptor computation: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000.0 << std::endl;
	std::cout << "/*size of issS = " << issS->size() << "; size of issT = " << issT->size() << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	/*========================correspondences matching=========================*/
	std::cout << "/*matching correspondences..." << std::endl;
	std::vector<int> corr_NOS, corr_NOT;
	GrorPre::correspondenceSearching(fpfhS, fpfhT, *corr, max_corr, corr_NOS, corr_NOT);
	auto t4 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of matching correspondences: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000.0 << std::endl;
	std::cout << "/*number of correspondences= " << corr->size() << std::endl;
	std::cout << "/*=================================================*/" << std::endl;
}

void GrorPre::grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, Eigen::Vector3f &centerS, Eigen::Vector3f &centerT, pcl::CorrespondencesPtr corr, double resolution)
{
	int max_corr = 5;// neighbor number in descriptor searching
	auto t = std::chrono::system_clock::now();
	/*=============down sample point cloud by voxel grid filter=================*/
	std::cout << "/*voxel grid sampling......" << resolution << std::endl;
	GrorPre::voxelGridFilter(origin_cloudS, cloudS, resolution);
	GrorPre::voxelGridFilter(origin_cloudT, cloudT, resolution);

	auto t1 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of cloud down sample : " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	/*=========================extract iss key points===========================*/
	std::cout << "/*extracting ISS keypoints......" << std::endl;
	pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);
	pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);
	GrorPre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);
	GrorPre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);
	auto t2 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of iss key point extraction: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
	std::cout << "/*=================================================*/" << std::endl;
	//translating the center of both point clouds to the origin
	pcl::PointXYZ ps, pt;
	pcl::computeCentroid(*issS, ps);

	for (int i = 0; i < issS->size(); i++) {
		issS->points[i].x -= ps.x;
		issS->points[i].y -= ps.y;
		issS->points[i].z -= ps.z;
	}
	pcl::computeCentroid(*issT, pt);
	for (int i = 0; i < issT->size(); i++) {
		issT->points[i].x -= pt.x;
		issT->points[i].y -= pt.y;
		issT->points[i].z -= pt.z;
	}
	centerS =Eigen::Vector3f(ps.x, ps.x, ps.x);
	centerT = Eigen::Vector3f(pt.x, pt.x, pt.x);
	/*======================fpfh descriptor computation=========================*/
	std::cout << "/*fpfh descriptor computation......" << std::endl;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
	GrorPre::fpfhComputation(cloudS, resolution, iss_IdxS, fpfhS);
	GrorPre::fpfhComputation(cloudT, resolution, iss_IdxT, fpfhT);
	auto t3 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of fpfh descriptor computation: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000.0 << std::endl;
	std::cout << "/*size of issS = " << issS->size() << "; size of issT = " << issT->size() << std::endl;
	std::cout << "/*=================================================*/" << std::endl;

	/*========================correspondences matching=========================*/
	std::cout << "/*matching correspondences..." << std::endl;
	std::vector<int> corr_NOS, corr_NOT;
	GrorPre::correspondenceSearching(fpfhS, fpfhT, *corr, max_corr, corr_NOS, corr_NOT);
	auto t4 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of matching correspondences: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000.0 << std::endl;
	std::cout << "/*number of correspondences= " << corr->size() << std::endl;
	std::cout << "/*=================================================*/" << std::endl;
}

void GrorPre::centroidTransMatCompute(Eigen::Matrix4f &T, const Eigen::Vector3f &vS, const Eigen::Vector3f &vT){
	Eigen::Vector3f t = T.block(0, 3, 3, 1);
	Eigen::Matrix3f R = T.block(0, 0, 3, 3);

	Eigen::Transform<float, 3, Eigen::Affine> a3f_truth(R);
	Eigen::Vector3f centerSt(0,0,0);
	pcl::transformPoint(vS, centerSt, a3f_truth);

	t = t - vT + centerSt;

	T.block(0, 3, 3, 1) = t;
}
