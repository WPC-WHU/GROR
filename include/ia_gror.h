#pragma once
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

#include <pcl/common/common.h>
#include <pcl/common/geometry.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/matching_candidate.h>

#define DUMMY_PRECISION 1e-12

namespace pcl {
	namespace registration
	{
		template <typename PointSource, typename PointTarget, typename Scalar>
		class GRORInitialAlignment : public Registration<PointSource, PointTarget>
		{
		public:
			using Registration<PointSource, PointTarget>::reg_name_;
			using Registration<PointSource, PointTarget>::input_;
			using Registration<PointSource, PointTarget>::indices_;
			using Registration<PointSource, PointTarget>::target_;
			using Registration<PointSource, PointTarget>::final_transformation_;
			using Registration<PointSource, PointTarget>::transformation_;
			using Registration<PointSource, PointTarget>::corr_dist_threshold_;
			using Registration<PointSource, PointTarget>::min_number_correspondences_;
			using Registration<PointSource, PointTarget>::tree_;
			using Registration<PointSource, PointTarget>::transformation_estimation_;
			using Registration<PointSource, PointTarget>::getClassName;

			typedef typename Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
			typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
			typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

			typedef typename Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
			typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
			typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

			typedef PointIndices::Ptr PointIndicesPtr;
			typedef PointIndices::ConstPtr PointIndicesConstPtr;


			typedef boost::shared_ptr<GRORInitialAlignment<PointSource, PointTarget, Scalar> > Ptr;
			typedef boost::shared_ptr<const GRORInitialAlignment<PointSource, PointTarget, Scalar> > ConstPtr;

			/** \Rotation Element including rotation axis and begin point of axis vector*/
			struct  RotationElement{
				/** \rotation axis. */
				Eigen::Vector3f rot_axis;
				/** \origin point of rotation. */
				Eigen::Vector3f rot_origin;
			};

			/** \brief intersection interval of a circular arc and an epsilon-ball, used for doing sweep algorithm*/
			struct IntervalEnd{
				/** \location of the end point. */
				double location;
				/** \is this end point a starting point of an interval. */
				bool is_start;
				/** \correspondence index. */
				int corr_idx; 

				void formIntervalEnd(const double &location_in, const bool &is_start_in, const int &corr_idx_in) {
					location = location_in;
					is_start = is_start_in;
					corr_idx = corr_idx_in;
				}
			};

			typedef typename KdTreeFLANN<Scalar>::Ptr FeatureKdTreePtr;

			/** \brief Constructor. */
			GRORInitialAlignment() : best_count_(3)
			{
				reg_name_ = "GRORInitialAlignment";
				key_source_ = PointCloudSource().makeShared();
				key_target_ = PointCloudSource().makeShared();
			};


			/** \brief set the down sample size (resolution of point cloud)
			* \param The resolution of point cloud
			*/
			void
				setResolution(float resolution) {
				resolution_ = resolution;
			};

			/** \brief Get the resolution of point cloud, as set by the user 
			*/
			float
				getResolution() {
				return resolution_;
			};
			
			/** \brief Set the number of used threads if OpenMP is activated.
			* \param[in] nr_threads the number of used threads
			*/
			inline void
				setNumberOfThreads(int nr_threads)
			{
				nr_threads_ = nr_threads;
			};

			/** \return the number of threads used if OpenMP is activated. 
			*/
			inline int
				getNumberOfThreads() const
			{
				return (nr_threads_);
			};

			/** \brief set the initial best_cout_ (3 default)
			* \param The initial best_cout_
			*/
			void
				setBestCount(int best_count) {
				best_count_ = best_count;
			};

			/** \brief Get the final best count(The number of maximum consistency set) of registration .
			*/
			int
				getBestCount() {
				return best_count_;
			};

			/** \brief Provide a pointer to the vector of the input correspondences.
			* \param[in] correspondences the boost shared pointer to a correspondence vector
			*/
			void
				setInputCorrespondences(const CorrespondencesPtr &correspondences);

			/** \brief Provide a pointer to the vector of the inlier correspondences.
			* \param[in] correspondences the boost shared pointer to a correspondence vector
			*/
			void
				getInlierCorrespondences(CorrespondencesPtr &correspondences) { correspondences = inliers_; };

			/** \brief Set the optimal selection number
			* \param n_optimal the optimal selection number
			*/
			void
				setOptimalSelectionNumber(int K_optimal) { K_optimal_ = K_optimal; };


		protected:

			/** \brief After the optimal selection step, delete the redundant points and reorganize the correspondence.
			* \param[in/out] key_source  a cloud of source point
			* \param[in/out] key_target  a cloud of target point
			* \param[in/out] correspondences correspondences points
			*/
			void 
				clearReduentPoints(PointCloudSource & key_source, PointCloudTarget & key_target, Correspondences & correspondences);

			/** \brief re-build correspondences graph, enumerate all pair of correspondences.
			* \param[out] corr_graph  Store  correspondences graph that meet the geometric constraints.
			*/
			void
				enumeratePairOfCorrespondence(std::vector<std::vector<std::array<Correspondence, 2>>> &corr_graph);

			/** \brief Potential unreliable correspondences removal Based On Edge Voting strategy and select fixed N optimal points for Maximum Consistent Set step.
			*\param[in] input correspondences
			*\param[out] output correspondences
			*\param[out] The fixed number of correspondence points for optimal selection.
			*/
			void 
				optimalSelectionBasedOnNodeReliability(Correspondences &input_correspondences, Correspondences &output_correspondences, const int K_optimal);

			/** \brief obtain the Maximum Consistent Set of correspondences
			*\param[in] correspondences  a vector store correspondences.
			*\param[in] corr_graph  correspondences graph information.
			*\param[out] two_point_tran_mat  a matrix for align two point correspondences.
			*\param[out] two_point_rot_ele  rotation element for aligned two point correspondences.
			*\param[out] best_angle  The best rotation angle witch takes two_point_rot_ele.rot_axis as the rotation axis for all correspondences.
			*/
			void 	
				obtainMaximumConsistentSetBasedOnEdgeReliability(Correspondences & correspondences, std::vector<std::vector<std::array<Correspondence, 2>>> corr_graph, Eigen::Matrix4f &two_point_tran_mat,RotationElement &two_point_rot_ele, float &best_angle);


			/** \brief recalculate the transformation matrix by the  maximum consistency set.
			*\param[out] transform The computed transformation.
			*/
			Eigen::Matrix4f
				refineTransformationMatrix(Eigen::Matrix4f & transform);

			/** \brief Rigid transformation computation method.
			* \param output the transformed input point cloud dataset using the rigid transformation found
			* \param guess The computed transformation
			*/
			virtual void
				computeTransformation(PointCloudSource &output, const Eigen::Matrix4f& guess);


			/** \brief The fixed number of correspondence points for optimal selection. */
			int K_optimal_;


			/** \brief The resolution of point cloud. */
			float resolution_;

			/** \brief Number of threads for parallelization (standard = 1).
			* \note Only used if run compiled with OpenMP.
			*/
			int nr_threads_;

			/** \brief  The number of maximum consistency point set.*/
			int best_count_;

			/** \brief The input correspondences. */
			CorrespondencesPtr input_correspondences_;

			/** \brief The output correspondences. */
			CorrespondencesPtr output_correspondences_;

			/** \brief The key point cloud in source after outliers removal. */
			PointCloudSourcePtr key_source_;

			/** \brief The key point cloud in target after outliers removal. */
			PointCloudTargetPtr key_target_;

			std::vector<int> remain_source_index_;

			std::vector<int> remain_target_index_;

			/** \brief The output correspondences. */
			CorrespondencesPtr inliers_;

			/** \brief The correspondences graph, each elements store a graph construct by a corresponding points and the other corresponding points that meet the geometric constraints after translation.*/
			std::vector<std::vector<std::array<Correspondence, 2>>> corr_graph_;

			


		private:

			/** \brief two pair correspondences points align.
			* \param[in] first_t target point in the first correspondence
			* \param[in] first_s source point in the first correspondence
			* \param[in] second_t target point in the second correspondence
			* \param[in] second_s source point in the second correspondence
			* \param[out] rot_element The rotation element for two correspondences
			* \return rotation matrix for two pair correspondences points align
			*/
			Eigen::Matrix4f
				twoPairPointsAlign(PointTarget first_t, PointSource first_s, PointTarget second_t, PointSource second_s, RotationElement &rot_element);

			/** \brief two vectors align.
			* \param[in] a The first vector
			* \param[in] b The second vector
			* \return Rotation Matrix of two vector 
			*/
			Eigen::Matrix3f
				twoVectorsAlign(const Eigen::Vector3f & a, const Eigen::Vector3f & b);

			Eigen::Matrix3f
				SkewSymmetric(Eigen::Vector3f in);

			/** \brief Calculate the Edge Reliability In Relaxed Constraint Function Space(RCFS)
			* \param[in] mat rotation matrix for edge pair correspondences align
			* \param[in] rotation_element rotation element for two correspondences align
			* \param[in] diff_to_s source points after translate to source origin
			* \param[in] diff_to_t target points after translate to target origin
			* \return degree of the edge reliability in (RCFS)
			*/
			int
				calEdgeReliabilityInRCFS(Eigen::Matrix4f &mat, RotationElement &rotation_element, std::vector<Eigen::Vector3f> &diff_to_s, std::vector<Eigen::Vector3f> &diff_to_t);


			/** \brief  Calculate the Edge Reliability In Tight Constraint Function Space(RCFS)
			* \param[in] transform rotation matrix for two pair correspondences points align
			* \param[in] rotation_element rotation element for two pair correspondences points align
			* \return 1D rotation angle and max count(lower bound)
			*/
			std::tuple<float, int>
				calEdgeReliabilityInTCFS(Eigen::Matrix4f & transform, RotationElement & rotation_element);

			inline 
				float vl_fast_atan2_f(float y, float x)
			{
				float angle, r;
				float const c3 = 0.1821F;
				float const c1 = 0.9675F;
				float abs_y = std::abs(y);

				if (x >= 0)
				{
					r = (x - abs_y) / (x + abs_y);
					angle = (float)(3.1415926f / 4);
				}
				else
				{
					r = (x + abs_y) / (abs_y - x);
					angle = (float)(3 * 3.1415926f / 4);
				}
				angle += (c3*r*r - c1) * r;
				return (y < 0) ? -angle : angle;
			}


			void
				removeOutlierPair(CorrespondencesPtr pair_info_ptr, int index);

			/** \brief insert angular intervals to a vector.
			* \param[out] interval_arr a vector store angular intervals
			* \param[in] start_pt start angular(pt) for an interval
			* \param[in] start_pt end   angular(pt) for an interval
			* \param[in] corr_idx correspondence index
			*/
			void
				insertInterval(std::vector<IntervalEnd> &interval_arr, const double &start_pt, const double &end_pt, const int &corr_idx);

			/** \brief calculate the circle intersection.
			*/
			double 
				circleIntersection(double R, double d, double r);

			/** \brief  The max-stabbing problem aims to find a vertical line that "stabs" the maximum number of intervals.
			* \param[in] interval_array a vector store angular intervals
			* \param[out] out_angle stab the out angle
			* \param[in] one_to_one is one to one
			*/
			void 
				intervalStab(std::vector<IntervalEnd> &interval_array, double &out_angle, int &out_upbnd, bool one_to_one);
		};

	}
}

#include"ia_gror.hpp"