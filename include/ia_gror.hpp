#pragma once

#include "ia_gror.h"
#include<omp.h>

bool sortByVoteNumber(const std::pair<int, int> a, const std::pair<int, int> b) {
	return a.second > b.second;
};

template <typename T>
bool sortByNumber(const T &a, const T &b) {
	return a.size() > b.size();
};

template <class T>
bool compareIntervalEnd(const T &intA, const T &intB)
{
	return (intA.location < intB.location);
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::setInputCorrespondences(const CorrespondencesPtr & correspondences)
{
	input_correspondences_.reset(new Correspondences);
	input_correspondences_ = correspondences;
};


template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::clearReduentPoints(PointCloudSource & key_source, PointCloudTarget & key_target, Correspondences & correspondences)
{
	std::vector<bool> v_list_t(key_target.size(), false);
	std::vector<bool> v_list_s(key_source.size(), false);

	PointCloudSource key_target_temp;
	PointCloudSource key_source_temp;

	for (int i = 0; i<correspondences.size(); ++i) {
		v_list_t[correspondences[i].index_match] = true;
		v_list_s[correspondences[i].index_query] = true;
	}

	for (int i = 0; i < v_list_t.size(); ++i) {
		if (v_list_t[i]) {
			remain_target_index_.push_back(i);
			key_target_temp.push_back((*target_)[i]);
		}
	}

	std::map<int, int> look_table;

	for (int i = 0; i < remain_target_index_.size(); ++i) {
		look_table.insert({ remain_target_index_[i],i });
	}

	for (int i = 0; i<correspondences.size(); ++i) {
		correspondences[i].index_match = look_table[correspondences[i].index_match];
	}

	look_table.clear();


	for (int i = 0; i < v_list_s.size(); ++i) {
		if (v_list_s[i]) {
			remain_source_index_.push_back(i);
			key_source_temp.push_back((*input_)[i]);
		}
	}

	for (int i = 0; i < remain_source_index_.size(); ++i) {
		look_table.insert({ remain_source_index_[i],i });
	}

	for (int i = 0; i < correspondences.size(); ++i) {
		correspondences[i].index_query = look_table[correspondences[i].index_query];
	}
	key_source = key_source_temp;
	key_target = key_target_temp;
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::enumeratePairOfCorrespondence(std::vector<std::vector<std::array<Correspondence, 2>>> &corr_graph)
{
	int corr_size = output_correspondences_->size();

	corr_graph.resize(corr_size);

	for (int i = 0; i < corr_size; ++i) {

		std::vector<std::array<Correspondence, 2>> i_pair_vec;

		for (int j = 1; j < corr_size; ++j)
		{
			if (i >= j) {
				continue;
			}

			auto first_corr = (*output_correspondences_)[i];
			auto second_corr = (*output_correspondences_)[j];

			auto first_corr_s = (*key_source_)[first_corr.index_query];
			auto first_corr_t = (*key_target_)[first_corr.index_match];

			auto second_corr_s = (*key_source_)[second_corr.index_query];
			auto second_corr_t = (*key_target_)[second_corr.index_match];

			float dis_1 = pcl::geometry::distance(first_corr_s, second_corr_s);
			float dis_2 = pcl::geometry::distance(first_corr_t, second_corr_t);

			float delta_dis = std::abs(dis_1 - dis_2);

			if (delta_dis < (2.0*resolution_)/* && dis_1 >(5.0*resolution_)*/)
			{
				std::array<Correspondence, 2> tmp;
				tmp[0] = (*output_correspondences_)[i];
				tmp[1] = (*output_correspondences_)[j];

				i_pair_vec.emplace_back(tmp);
			}

		}
		corr_graph[i] = i_pair_vec;
	}
}
template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::optimalSelectionBasedOnNodeReliability(Correspondences & input_correspondences, Correspondences & output_correspondences, int K_optimal)
{
	int corr_size = input_correspondences.size();

	std::pair<int, int>init_p(0, 0);

	//We directly sum the Adjacent Matrix of each node without list A
	std::vector<std::pair<int, int>> node_degree(corr_size, init_p);


	//#ifdef _OPENMP
	//#pragma omp parallel for shared (node_degree) num_threads(nr_threads_)
	//#endif
	for (int i = 0; i < corr_size; i++) {
		node_degree[i].first = i;
		int id_s1 = input_correspondences[i].index_query;
		int id_t1 = input_correspondences[i].index_match;

		PointSource p_s1 = (*input_).points[id_s1];
		PointTarget p_t1 = target_->points[id_t1];

		for (int j = 0; j < corr_size; j++) {

			if (i >= j) {
				continue;
			}

			int id_s2 = input_correspondences[j].index_query;
			int id_t2 = input_correspondences[j].index_match;

			PointSource p_s2 = (*input_).points[id_s2];
			PointTarget p_t2 = target_->points[id_t2];

			float edge_dis_s = pcl::geometry::distance(p_s1, p_s2);
			float edge_dis_t = pcl::geometry::distance(p_t1, p_t2);

			float delta_dis = std::abs(edge_dis_s - edge_dis_t);

			if (delta_dis < 2.0 * resolution_) {
				node_degree[i].second++;
				node_degree[j].second++;
			}
		}
	}

	//Sort points according note number 
	//Points with higher degree have higher reliability
	sort(node_degree.begin(), node_degree.end(), sortByVoteNumber);

	Correspondences corr_tmp(K_optimal);
	if (corr_size >= K_optimal) {
		for (int i = 0; i < K_optimal; i++) {
			int id = node_degree[i].first;
			corr_tmp[i] = input_correspondences[id];
		}
		output_correspondences = corr_tmp;
	}
	else {
		output_correspondences = input_correspondences;
	}
	*key_source_ = *input_;
	*key_target_ = *target_;
	clearReduentPoints(*key_source_, *key_target_, output_correspondences);

	//oblation expriment
	//output_correspondences = input_correspondences;
	//*key_source_ = *input_;
	//*key_target_ = *target_;
}

//each edge pair : E_ij^ST = (e_ij^S, e_ij^T), e_ij^S = (v_i^S, v_j^S), e_ij^T = (v_i^T, v_j^T).
//remain node pairs accompany with each edge pair E_ij^ST :V_k = (v_k^S, v_k^T), k != i != j
template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::obtainMaximumConsistentSetBasedOnEdgeReliability(Correspondences & correspondences, std::vector<std::vector<std::array<Correspondence, 2>>> corr_graph, Eigen::Matrix4f &two_point_tran_mat, RotationElement &two_point_rot_ele, float &best_angle)
{

	std::sort(corr_graph.begin(), corr_graph.end(), sortByNumber<std::vector<std::array<Correspondence, 2>>>);
	//#pragma omp parallel for num_threads(nr_threads_)
	for (int i = 0; i < corr_graph.size(); i++) {
		if (corr_graph[i].size()<10) {
			continue;
		}

		Correspondence first_pair = corr_graph[i][0][0];//first node pair of the edge pair

		auto first_corr_s = (*key_source_)[first_pair.index_query];// first node of the source edge 
		auto first_corr_t = (*key_target_)[first_pair.index_match];// first node of the target edge 

		Correspondence second_pair = corr_graph[i][0][1];//first node pair of the edge pair

		auto second_corr_s = (*key_source_)[second_pair.index_query];// first node of the source edge 
		auto second_corr_t = (*key_target_)[second_pair.index_match];// first node of the target edge 

		std::vector<Eigen::Vector3f> diff_to_s(correspondences.size());
		std::vector<Eigen::Vector3f> diff_to_t(correspondences.size());

		for (int j = 0; j < correspondences.size(); ++j)
		{
			auto& t_p = (*key_target_)[correspondences[j].index_match];
			auto& s_p = (*key_source_)[correspondences[j].index_query];

			diff_to_t[j] = t_p.getVector3fMap() - first_corr_t.getVector3fMap();
			diff_to_s[j] = s_p.getVector3fMap() - first_corr_s.getVector3fMap();
		}

		RotationElement rotation_element;

		//align two edge
		Eigen::Matrix4f mat = twoPairPointsAlign(first_corr_t, first_corr_s, second_corr_t, second_corr_s, rotation_element);

		//degree of edge reliability(der) in relaxed constraint function space(rcfs)
		int der_in_rcfs = calEdgeReliabilityInRCFS(mat, rotation_element, diff_to_s, diff_to_t);


		if (der_in_rcfs <= best_count_) {
			continue;
		}

		auto result = calEdgeReliabilityInTCFS(mat, rotation_element);

		float angle = std::get<0>(result);//last freedom theta

		//degree of edge reliability(der) in tight constraint function space(tcfs)
		int der_in_tcfs = std::get<1>(result);

		if (der_in_tcfs > best_count_)
		{
			best_count_ = der_in_tcfs;
			two_point_rot_ele = rotation_element;
			two_point_tran_mat = mat;
			best_angle = angle;
		}
	}
};
template<typename PointSource, typename PointTarget, typename Scalar>
inline Eigen::Matrix4f pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::refineTransformationMatrix(Eigen::Matrix4f & transform)
{
	PointCloudSourcePtr gr_issS(new PointCloudSource);
	inliers_.reset(new Correspondences);
	pcl::transformPointCloud(*input_, *gr_issS, transform);
	Eigen::Matrix4f p_p_matrix = Eigen::Matrix4f::Identity();
	int sum = 0;
	std::vector<int> corr_p_s_index;
	std::vector<int> corr_p_t_index;
	std::vector<double> v_dists;


	for (int i = 0; i < input_correspondences_->size(); i++)
	{
		int idxS = (*input_correspondences_)[i].index_query;
		int idxT = (*input_correspondences_)[i].index_match;

		auto t_p = (*target_)[idxT];
		auto s_p = (*gr_issS)[idxS];

		float dis = pcl::geometry::distance(t_p, s_p);

		if (dis < 2 * resolution_)
		{
			corr_p_s_index.push_back(idxS);
			corr_p_t_index.push_back(idxT);
			inliers_->push_back((*input_correspondences_)[i]);
			v_dists.push_back(dis);
			sum++;
		}
	}

	bool use_umeyama_ = true;
	if (use_umeyama_)
	{
		int N = corr_p_t_index.size();
		Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_src(3, N);
		Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_tgt(3, N);

		for (int i = 0; i < N; ++i)
		{
			cloud_src(0, i) = (*input_)[corr_p_s_index[i]].x;
			cloud_src(1, i) = (*input_)[corr_p_s_index[i]].y;
			cloud_src(2, i) = (*input_)[corr_p_s_index[i]].z;
			//++source_it;

			cloud_tgt(0, i) = (*target_)[corr_p_t_index[i]].x;
			cloud_tgt(1, i) = (*target_)[corr_p_t_index[i]].y;
			cloud_tgt(2, i) = (*target_)[corr_p_t_index[i]].z;
			//++target_it;
		}

		// Call Umeyama directly from Eigen (PCL patched version until Eigen is released)
		p_p_matrix = pcl::umeyama(cloud_src, cloud_tgt, false);

	}
	else {
		Eigen::Vector3f mc_t = Eigen::Vector3f::Zero();
		Eigen::Vector3f mc_s = Eigen::Vector3f::Zero();

		int N = corr_p_t_index.size();

		for (int i = 0; i < N; ++i)
		{
			auto p_t = (*target_)[corr_p_t_index[i]];
			auto p_s = (*input_)[corr_p_s_index[i]];

			mc_t += Eigen::Vector3f(p_t.x, p_t.y, p_t.z);
			mc_s += Eigen::Vector3f(p_s.x, p_s.y, p_s.z);
		}

		mc_t /= static_cast<double>(N);
		mc_s /= static_cast<double>(N);

		std::vector<Eigen::Vector3f> pc_t(corr_p_t_index.size());
		std::vector<Eigen::Vector3f> pc_s(corr_p_t_index.size());

		Eigen::Matrix3f W = Eigen::Matrix3f::Zero();

		for (int i = 0; i < N; ++i)
		{
			auto p_t = (*target_)[corr_p_t_index[i]];
			auto p_s = (*input_)[corr_p_s_index[i]];

			double weight = 1.0 / (v_dists[i] * 1.0);

			pc_t[i] = (Eigen::Vector3f(p_t.x, p_t.y, p_t.z) - mc_t)*weight;
			pc_s[i] = (Eigen::Vector3f(p_s.x, p_s.y, p_s.z) - mc_s)*weight;

			W += pc_t[i] * (pc_s[i].transpose());
		}

		Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

		Eigen::Matrix3f U = svd.matrixU();
		Eigen::Matrix3f V = svd.matrixV();

		Eigen::Matrix3f R = U*(V.transpose());
		Eigen::Vector3f t = mc_t - R*mc_s;

		p_p_matrix.block<3, 3>(0, 0) = R;
		p_p_matrix.block<3, 1>(0, 3) = t;
	}
	return p_p_matrix;
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::computeTransformation(PointCloudSource & output, const Eigen::Matrix4f & guess)
{
	// Point cloud containing the correspondences of each point in <input, indices>
	PointCloudSourcePtr input_transformed(new PointCloudSource);

	// Initialise final transformation to the guessed one
	final_transformation_ = guess;

	// If the guessed transformation is non identity
	if (guess != Eigen::Matrix4f::Identity())
	{
		input_transformed->resize(input_->size());
		// Apply guessed transformation prior to search for neighbours
		pcl::transformPointCloud(*input_, *input_transformed, guess);
	}
	else
		*input_transformed = *input_;

	output_correspondences_.reset(new Correspondences);
	auto t = std::chrono::system_clock::now();
	optimalSelectionBasedOnNodeReliability(*input_correspondences_, *output_correspondences_, K_optimal_);
	auto t1 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of optimalbSelectionBasedOnNodeReliability: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;

	enumeratePairOfCorrespondence(corr_graph_);
	auto t2 = std::chrono::system_clock::now();
	//std::cout << "/*Down!: time consumption of enumeratePairOfCorrespondence: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;

	Eigen::Matrix4f two_point_tran_mat;
	RotationElement two_point_rot_ele;
	float best_angle = 0.0;

	obtainMaximumConsistentSetBasedOnEdgeReliability(*output_correspondences_, corr_graph_, two_point_tran_mat, two_point_rot_ele, best_angle);
	auto t3 = std::chrono::system_clock::now();
	std::cout << "/*Down!: time consumption of obtainMaximumConsistentSet: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000.0 << std::endl;


	Eigen::Matrix4f IdM_1 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f IdM_2 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f IdM_3 = Eigen::Matrix4f::Identity();
	Eigen::Matrix3f rot = Eigen::AngleAxisf(best_angle, two_point_rot_ele.rot_axis).toRotationMatrix();
	IdM_1.block<3, 1>(0, 3) = -1.0*two_point_rot_ele.rot_origin;
	IdM_2.block<3, 3>(0, 0) = rot;
	IdM_3.block<3, 1>(0, 3) = two_point_rot_ele.rot_origin;
	Eigen::Matrix4f gr_tran_mat = IdM_3*IdM_2*IdM_1*two_point_tran_mat;
	final_transformation_ = refineTransformationMatrix(gr_tran_mat);

}
//https://math.stackexchange.com/questions/180418/
template<typename PointSource, typename PointTarget, typename Scalar>
inline Eigen::Matrix4f pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::twoPairPointsAlign(PointTarget first_t, PointSource first_s, PointTarget second_t, PointSource second_s, RotationElement &rot_element)
{

	Eigen::Vector3f vec_first_t(first_t.x, first_t.y, first_t.z), vec_first_s(first_s.x, first_s.y, first_s.z),
		vec_second_t(second_t.x, second_t.y, second_t.z), vec_second_s(second_s.x, second_s.y, second_s.z);
	Eigen::Vector3f vec_sour = (vec_first_s - vec_second_s).normalized();
	Eigen::Vector3f vec_tart = (vec_first_t - vec_second_t).normalized();

	rot_element.rot_axis = vec_tart;

	Eigen::Matrix3f rotation_mat = twoVectorsAlign(vec_sour, vec_tart);

	Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
	trans_mat.block<3, 3>(0, 0) = rotation_mat;

	Eigen::Vector3f translation_v1 = vec_first_t - rotation_mat * vec_first_s;
	Eigen::Vector3f translation_v2 = vec_second_t - rotation_mat * vec_second_s;

	trans_mat.block<3, 1>(0, 3) = 0.5*(translation_v1 + translation_v2);

	rot_element.rot_origin = vec_first_t;

	return  trans_mat.cast<float>();
}
template<typename PointSource, typename PointTarget, typename Scalar>
inline Eigen::Matrix3f pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::twoVectorsAlign(const Eigen::Vector3f & a, const Eigen::Vector3f & b)
{
	Eigen::Vector3f v = a.cross(b);

	float s = v.norm();
	float c = a.dot(b);

	Eigen::Matrix3f rotation_mat = Eigen::Matrix3f::Identity();

	Eigen::Matrix3f v_skew = SkewSymmetric(v);

	rotation_mat = rotation_mat + v_skew + v_skew*v_skew*(1.0f / (1.0f + c));

	return rotation_mat;
}
template<typename PointSource, typename PointTarget, typename Scalar>
inline Eigen::Matrix3f pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::SkewSymmetric(Eigen::Vector3f in)
{
	Eigen::Matrix3f Skew = Eigen::Matrix3f::Zero();

	Skew(0, 1) = -1.0f * in[2];
	Skew(0, 2) = in[1];
	Skew(1, 0) = in[2];
	Skew(1, 2) = -1.0f * in[0];
	Skew(2, 0) = -1.0f * in[1];
	Skew(2, 1) = in[0];

	return Skew;
}
template<typename PointSource, typename PointTarget, typename Scalar>
inline int pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::calEdgeReliabilityInRCFS(Eigen::Matrix4f & mat, RotationElement & rotation_element, std::vector<Eigen::Vector3f>& diff_to_s, std::vector<Eigen::Vector3f>& diff_to_t)
{
	Eigen::Transform<float, 3, Eigen::Affine> transform = Eigen::Transform<float, 3, Eigen::Affine>(mat);
	int sum = 0;//We directly sum the edge-node Affinity Matrix without list M

	auto &rot_origin = rotation_element.rot_origin;
	auto &rot_axis_t = rotation_element.rot_axis;
	Eigen::Vector3f rot_axis_s = transform.rotation().inverse()*rot_axis_t;

	//#ifdef _OPENMP
	//#pragma omp parallel for reduction(+:sum) num_threads(nr_threads_)
	//#endif
	for (int i = 0; i < output_correspondences_->size(); i++)
	{
		Eigen::Vector3f& delta_z_t = diff_to_t[i];
		Eigen::Vector3f& delta_z_s = diff_to_s[i];

		float dis_t = (delta_z_t).norm();
		float dis_s = (delta_z_s).norm();


		if (std::abs(dis_t - dis_s) < 2 * resolution_
			&&std::abs(delta_z_t.dot(rot_axis_t) - delta_z_s.dot(rot_axis_s)) < 2 * resolution_)
		{
			sum++;
		}
	}
	return sum;
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::removeOutlierPair(CorrespondencesPtr pair_info_ptr, int index)
{
	auto iter = pair_info_ptr->begin();
	pair_info_ptr->erase(iter + index);
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::insertInterval(std::vector<IntervalEnd>& interval_arr, const double & start_pt, const double & end_pt, const int & corr_idx)
{
	IntervalEnd int_end_tmp;
	int_end_tmp.formIntervalEnd(start_pt, true, corr_idx);
	interval_arr.push_back(int_end_tmp);
	int_end_tmp.formIntervalEnd(end_pt, false, corr_idx);
	interval_arr.push_back(int_end_tmp);
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline double pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::circleIntersection(double R, double d, double r)
{
	assert(R >= 0 && d >= 0 && r >= 0 && "parameters must be positive");


	//assert(d<(R+r));
	// Return value is between 0 and pi.

	double rat, x, angle;

	if (d <= DUMMY_PRECISION)
	{
		return M_PI;
	}

	//    if( fabs(d-(R+r))<DUMMY_PRECISION )
	//    {
	//        return 0;
	//    }

	x = (d*d - r*r + R*R) / (2 * d);

	rat = x / R;
	if (rat <= -1.0)
	{
		return M_PI;
	}

	angle = acos(rat);
	assert(angle <= M_PI && "angle must be < PI");
	return angle;
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline void pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::intervalStab(std::vector<IntervalEnd>& interval_array, double & out_angle, int & out_upbnd, bool one_to_one)
{
	std::vector<int> ACTab(key_source_->size(), 0);
	int curr_upbnd = 0;
	out_upbnd = 0;
	//1. sort interval_array
	std::sort(interval_array.begin(), interval_array.end(), compareIntervalEnd<IntervalEnd>);
	double currLoc = 0;
	int NOEnd = 0;
	if (!one_to_one) {
		for (size_t i = 0; i < interval_array.size(); i++) {
			//is a starting point
			if (interval_array[i].is_start) {
				ACTab[interval_array[i].corr_idx]++;
				if (ACTab[interval_array[i].corr_idx] == 1) {
					curr_upbnd++;
					if (curr_upbnd > out_upbnd) {
						out_upbnd = curr_upbnd;
						out_angle = interval_array[i].location;
					}
				}
			}
			else {
				ACTab[interval_array[i].corr_idx]--;
				if (ACTab[interval_array[i].corr_idx] == 0) {
					NOEnd++;
				}
			}
			if (interval_array[i].location > currLoc) {
				curr_upbnd -= NOEnd;
				NOEnd = 0;
				if (currLoc == out_angle) {
					out_angle = (currLoc + interval_array[i].location) / 2;
				}
				currLoc = interval_array[i].location;
			}
		}
		curr_upbnd -= NOEnd;
	}
	else {
		for (size_t i = 0; i < interval_array.size(); i++) {
			//is a starting point
			if (interval_array[i].is_start) {
				ACTab[interval_array[i].corr_idx]++;
				curr_upbnd++;
				if (curr_upbnd > out_upbnd) {
					out_upbnd = curr_upbnd;
					out_angle = interval_array[i].location;
				}
			}
			else {
				ACTab[interval_array[i].corr_idx]--;
				NOEnd++;
			}
			if (interval_array[i].location > currLoc) {
				curr_upbnd -= NOEnd;
				NOEnd = 0;
				currLoc = interval_array[i].location;
			}
		}
		curr_upbnd -= NOEnd;
	}
}

template<typename PointSource, typename PointTarget, typename Scalar>
inline std::tuple<float, int> pcl::registration::GRORInitialAlignment<PointSource, PointTarget, Scalar>::calEdgeReliabilityInTCFS(Eigen::Matrix4f & transform, RotationElement & rotation_element)
{
	std::vector<IntervalEnd> interval_array;

	const auto axis = rotation_element.rot_axis;
	const auto origin = rotation_element.rot_origin;

	PointCloudSourcePtr source_local(new PointCloudSource);
	PointCloudTargetPtr target_local(new PointCloudTarget);

	Eigen::Matrix4f IdM_1 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f IdM_2 = Eigen::Matrix4f::Identity();

	Eigen::Vector3f z_axis = Eigen::Vector3f::UnitZ();
	IdM_1.block<3, 1>(0, 3) = -1.0 * origin;
	IdM_2.block<3, 3>(0, 0) = twoVectorsAlign(axis, z_axis);

	Eigen::Matrix4f local_tm_t = IdM_2*IdM_1;
	Eigen::Matrix4f local_tm_s = local_tm_t*transform.matrix();

	pcl::transformPointCloud(*key_source_, *source_local, local_tm_s);
	pcl::transformPointCloud(*key_target_, *target_local, local_tm_t);

	const size_t msize_total = source_local->size();
	const size_t bsize_total = target_local->size();

	// Norm of points on XY-plane.
	std::vector<float> M_len(msize_total);
	std::vector<float> B_len(bsize_total);

	// Z-coordinate of points.
	std::vector<float> M_z(msize_total);
	std::vector<float> B_z(bsize_total);

	// Azimuth of points.
	std::vector<float> M_azi(msize_total);
	std::vector<float> B_azi(bsize_total);

	for (int idx = 0; idx < msize_total; ++idx)
	{
		pcl::PointXYZ s_p = (*source_local)[idx];

		float p_x = s_p.x;
		float p_y = s_p.y;

		M_z[idx] = s_p.z;
		M_len[idx] = std::sqrt(p_x*p_x + p_y*p_y);
		M_azi[idx] = vl_fast_atan2_f(p_y, p_x);
	}

	for (int idx = 0; idx < bsize_total; ++idx)
	{
		pcl::PointXYZ t_p = (*target_local)[idx];

		float p_x = t_p.x;
		float p_y = t_p.y;

		B_z[idx] = t_p.z;
		B_len[idx] = std::sqrt(p_x*p_x + p_y*p_y);
		B_azi[idx] = vl_fast_atan2_f(p_y, p_x);
	}

	double dz, d, thMz, rth, dev, beg, end;
	double threshold_ = 2.0*resolution_;
	for (int i = 0; i < output_correspondences_->size(); i++)
	{
		int idxS = (*output_correspondences_)[i].index_query;
		int idxT = (*output_correspondences_)[i].index_match;

		dz = B_z[idxT] - M_z[idxS];

		d = B_len[idxT] - M_len[idxS];

		thMz = threshold_*threshold_ - dz*dz;

		float TWOPI = 2.0*M_PI;

		if (d*d <= thMz) {
			rth = sqrt(thMz);
			//insert the intersection interval to int_idxS

			//insert [0,2pi] if M is too short
			if (M_len[idxS] <= DUMMY_PRECISION)
			{
				insertInterval(interval_array, 0, TWOPI, idxS);
			}
			else
			{
				dev = circleIntersection(M_len[idxS], B_len[idxT], rth);

				if (std::fabs(dev - M_PI) <= DUMMY_PRECISION)
				{
					/*That could be improved by instead of insert adding 1 to the final quality*/
					insertInterval(interval_array, 0, TWOPI, idxS);
				}
				else
				{
					beg = std::fmod(B_azi[idxT] - dev - M_azi[idxS], TWOPI);
					if (beg < 0)
					{
						beg += TWOPI;
					}
					end = std::fmod(B_azi[idxT] + dev - M_azi[idxS], TWOPI);
					if (end < 0)
					{
						end += TWOPI;
					}
					if (end >= beg)
					{
						insertInterval(interval_array, beg, end, idxS);
					}
					else
					{
						insertInterval(interval_array, beg, TWOPI, idxS);
						insertInterval(interval_array, 0, end, idxS);
					}
				}
			}
		}
	}

	double out_angle;
	int out_count;

	intervalStab(interval_array, out_angle, out_count, true);

	return std::make_tuple(out_angle, out_count);
}

