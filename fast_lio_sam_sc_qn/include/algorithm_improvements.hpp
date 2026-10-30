#ifndef FAST_LIO_SAM_SC_QN_ALGORITHM_IMPROVEMENTS_HPP
#define FAST_LIO_SAM_SC_QN_ALGORITHM_IMPROVEMENTS_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace fast_lio_sam_sc_qn
{
// 定义常量 PI
inline constexpr double kPi = 3.14159265358979323846;
/**
 * @brief 判断是否需要创建新的关键帧
 * 
 * 采取“或”的逻辑：只要平移量或者旋转量任何一个超过了预设阈值，就认为需要新建关键帧。
 * 
 * @param translation 当前帧与上一个关键帧的平移距离 (通常单位：米)
 * @param rotation 当前帧与上一个关键帧的旋转角度 (通常单位：弧度)
 * @param translation_threshold 平移阈值
 * @param rotation_threshold 旋转阈值
 * @return true 需要创建关键帧
 * @return false 不需要创建关键帧
 */
inline bool shouldCreateKeyframe(const double translation,
                                 const double rotation,
                                 const double translation_threshold,
                                 const double rotation_threshold)
{
    return translation >= translation_threshold || rotation >= rotation_threshold;
}

/**
 * @brief 回环候选帧一致性校验器 (共识机制)
 * 在 SLAM 中，单帧的回环检测往往会因为环境相似或噪声产生误匹配（假阳性）。
 * 该类通过检查连续多帧的检测结果是否“具有相似的滞后间隔（lag）”来确认回环。
 * 如果连续几帧都检测到了类似位置的候选帧，说明这不是偶然的误检，从而增强回环检测的鲁棒性。
 * 回环检测的防误报机制
 */
class LoopCandidateConsensus  
{
public:
    /**
     * @param required_confirmations 确认回环所需的连续次数 (默认 2 次)需要连续确定回环几次
     * @param lag_tolerance 允许的滞后步数误差容忍度 (默认 1)
     */
    LoopCandidateConsensus(consint required_confirmations = 2,
                           const int lag_tolerance = 1)
        : required_confirmations_(std::max(1, required_confirmations)),
          lag_tolerance_(std::max(0, lag_tolerance))
    {
    }

    /**
     * @brief 更新校验器状态，输入当前帧索引和检测到的候选帧索引
     * @param query_idx 当前查询帧的索引
     * @param candidate_idx 匹配到的历史帧的索引，这一帧和历史第几帧匹配
     * @return int 如果确认是真实回环，返回 candidate_idx；否则返回 -1
     */
    int update(const int query_idx, const int candidate_idx)
    {
        // 合法性检查：候选帧必须是历史帧，且不能是当前帧自己
        if (candidate_idx < 0 || candidate_idx >= query_idx)
        {
            reset();
            return -1;
        }

        // 计算“时间滞后量”（当前帧和匹配历史帧之间的索引差）
        const int lag = query_idx - candidate_idx;
        
        // 判断是否还是同一个地方的候选：
        // 1. 必须有上一次的查询记录 (last_query_idx_ >= 0)
        // 2. 当前帧必须是前进的 (query_idx > last_query_idx_)
        // 3. 本次的滞后量 lag 与上一次的滞后量 last_lag_ 非常接近 (在容差范围内)
        const bool follows_same_place = last_query_idx_ >= 0 &&
            query_idx > last_query_idx_ &&
            std::abs(lag - last_lag_) <= lag_tolerance_;

        // 如果是一直在往同一个地方开，累加确认次数；如果断断续续，重新从1开始计
        confirmations_ = follows_same_place ? confirmations_ + 1 : 1;
        last_query_idx_ = query_idx;
        last_lag_ = lag;
        
        // 如果连续确认次数达到阈值，则输出成功的候选帧索引
        return confirmations_ >= required_confirmations_ ? candidate_idx : -1;
    }

    // 重置内部状态（通常用于轨迹发生重定位或大幅度跳跃时）
    void reset()
    {
        last_query_idx_ = -1;
        last_lag_ = 0;
        confirmations_ = 0;
    }
private:
    int required_confirmations_; // 需要连续确认的次数
    int lag_tolerance_;          // 允许的滞后量波动范围
    int last_query_idx_ = -1;    // 上一次查询的帧索引
    int last_lag_ = 0;           // 上一次的滞后量
    int confirmations_ = 0;      // 当前连续确认计数
};

/**
 * @brief 回环因子方差配置参数结构体
 */
struct LoopFactorVarianceConfig
{
    double fitness_sigma_scale = 1.0;                  // 拟合得分到标准差的缩放系数
    double translation_sigma_min = 0.03;               // 平移标准差下限 (米)
    double translation_sigma_max = 0.50;               // 平移标准差上限 (米)
    double rotation_sigma_min_rad = 0.5 * kPi / 180.0; // 旋转标准差下限 (弧度，默认 0.5度)
    double rotation_sigma_max_rad = 10.0 * kPi / 180.0;// 旋转标准差上限 (弧度，默认 10度)
};

/**
 * @brief 根据 ICP 等匹配算法的适应度得分（Fitness Score），动态计算回环因子的方差
 * 
 * 核心思想：
 * - Fitness Score 在这里被视为“均方距离”的代理指标。得分越低，说明两帧点云匹配得越好。
 * - 由于是距离的平方，所以求标准差时需要对 score 开根号。
 * - 假设平移误差和旋转误差存在一定的几何关系，利用目标点云的 RMS 半径，将平移标准差转换为旋转标准差。
 * - 使用 GTSAM 的 Pose3 切线空间排序：前三个是旋转 (Rotation)，后三个是平移 (Translation)。
 * 
 * @param fitness_score 点云配准的适应度得分（对应均方误差的平方）
 * @param target_rms_radius 目标点云的均方根半径 (用于关联平移和旋转误差)
 * @param config 方差配置参数
 * @return Eigen::Matrix<double, 6, 1> 6自由度方差向量 [Rx, Ry, Rz, Tx, Ty, Tz]^T
 */
inline Eigen::Matrix<double, 6, 1> loopFactorVariances(const double fitness_score,
                                                       const double target_rms_radius,
                                                       const LoopFactorVarianceConfig &config)
{
    // 1. 异常值处理（防止 NaN 或负数导致后续计算崩溃）
    const double safe_score = std::isfinite(fitness_score) ? std::max(0.0, fitness_score)
                                                           : std::numeric_limits<double>::infinity();
    
    // 2. 提取并检查平移标准差上下限
    const double translation_min = std::isfinite(config.translation_sigma_min)
        ? std::max(1e-4, config.translation_sigma_min)
        : 0.03;
    const double translation_max = std::isfinite(config.translation_sigma_max)
        ? std::max(translation_min, config.translation_sigma_max)
        : std::max(translation_min, 0.50);
    
    // 3. 提取缩放系数
    const double score_scale = std::isfinite(config.fitness_sigma_scale)
        ? std::max(0.0, config.fitness_sigma_scale)
        : 1.0;
    
    // 4. 计算平移标准差：基于 score 开根号 (因为 score 代表距离平方)，并进行截断限制
    const double sigma_translation = std::clamp(
        score_scale * std::sqrt(safe_score), translation_min, translation_max);
    
    // 5. 获取目标点云半径（用于把平移误差转化为角度误差，公式: 角度 = 弧长 / 半径）
    const double radius = std::isfinite(target_rms_radius)
        ? std::max(0.1, target_rms_radius) // 防止半径过小导致除零
        : 0.1;
        
    // 6. 提取并检查旋转标准差上下限
    const double rotation_min = std::isfinite(config.rotation_sigma_min_rad)
        ? std::max(1e-5, config.rotation_sigma_min_rad)
        : 0.5 * kPi / 180.0;
    const double rotation_max = std::isfinite(config.rotation_sigma_max_rad)
        ? std::max(rotation_min, config.rotation_sigma_max_rad)
        : std::max(rotation_min, 10.0 * kPi / 180.0);
        
    // 7. 计算旋转标准差：平移标准差除以半径，并进行截断限制
    const double sigma_rotation = std::clamp(sigma_translation / radius, rotation_min, rotation_max);

    // 8. 组装 6x1 的方差向量 (注意：方差是标准差的平方)
    Eigen::Matrix<double, 6, 1> variances;
    variances << sigma_rotation * sigma_rotation,     // Rx Variance
                 sigma_rotation * sigma_rotation,     // Ry Variance
                 sigma_rotation * sigma_rotation,     // Rz Variance
                 sigma_translation * sigma_translation, // Tx Variance
                 sigma_translation * sigma_translation, // Ty Variance
                 sigma_translation * sigma_translation; // Tz Variance
    return variances;
}

} // namespace fast_lio_sam_sc_qn

#endif