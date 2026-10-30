// =====================================================================
// fast_lio_sam_sc_qn_main.h
// 一句话：这是一个 "LiDAR SLAM 主节点" 的头文件。
//   它把三件事串起来：
//     ① 前端（FAST-LIO 提供的里程计位姿，就是本文件的输入之一）
//     ② 回环检测（Scan Context 认路 + Quatro 配准验证）
//     ③ 后端优化（GTSAM iSAM2 修正整条轨迹）
//   最终输出：修正后的位姿、轨迹、全局点云地图，并支持存成 bag/PCD/KITTI。
// =====================================================================
#ifndef FAST_LIO_SAM_SC_QN_MAIN_H
#define FAST_LIO_SAM_SC_QN_MAIN_H

///// ========== 一、通用 C++ 标准库头文件 ==========
#include <ctime>        // 时间函数（如 std::time）
#include <cmath>        // 数学函数（abs, sin 等）
#include <chrono>       // C++11 高精度计时（测某段代码耗时用）
#include <vector>       // 动态数组（存关键帧、回环对等）
#include <memory>       // 智能指针（std::shared_ptr / unique_ptr）
#include <deque>        // 双端队列
#include <mutex>        // 互斥锁（多线程保护共享数据用）
#include <string>       // 字符串
#include <utility>      // std::pair / std::make_pair（如回环索引对）
#include <tuple>        // 元组（多值组合）
#include <filesystem>   // C++17 文件系统操作（建目录/判断文件存在）
#include <fstream>      // 文件读写流（存地图/日志）
#include <iostream>     // 标准输入输出（print 调试）
#include <atomic>       // 原子变量（跨线程安全读写的 bool/int）

///// ========== 二、ROS 相关头文件 ==========
#include <ros/ros.h>                 // ROS 核心：节点/句柄/订阅/发布
#include <ros/package.h>             // 获取当前包的安装路径（get package_path）
#include <rosbag/bag.h>              // 把地图保存成 rosbag 文件用
#include <tf/LinearMath/Quaternion.h>// 四元数（用于 四元数<->欧拉角 转换）
#include <tf/LinearMath/Matrix3x3.h> // 3x3 旋转矩阵，配合四元数转欧拉角
#include <tf/transform_datatypes.h>  // 常用变换工具（如 createQuaternionFromRPY：由 翻滚/俯仰/偏航 造四元数）
#include <tf_conversions/tf_eigen.h> // tf <-> Eigen 矩阵互转（方便用 Eigen 做位姿运算）
#include <tf/transform_broadcaster.h>// 坐标变换广播器（把 map->odom 变换发出去）
#include <std_msgs/String.h>         // 字符串消息（接收"保存地图"命令用）
#include <geometry_msgs/PoseStamped.h> // 带时间戳的位姿消息（发布实时位姿用）
#include <visualization_msgs/Marker.h> // 在 RViz 里画线/画点/画圆柱（回环可视化用）
#include <message_filters/subscriber.h>        // 消息过滤器订阅器（可做时间同步）
#include <message_filters/time_synchronizer.h> // 时间同步器（精确同步）
#include <message_filters/sync_policies/approximate_time.h> // 近似时间同步策略

///// ========== 三、GTSAM（后端因子图优化库）相关头文件 ==========
#include <gtsam/geometry/Rot3.h>      // 3D 旋转（SO(3)）
#include <gtsam/geometry/Point3.h>    // 3D 点
#include <gtsam/geometry/Pose3.h>     // 3D 位姿（旋转+平移）
#include <gtsam/slam/PriorFactor.h>   // "先验因子"：把某个位姿钉死在参考系里（比如第0帧 = 原点）
#include <gtsam/slam/BetweenFactor.h> // "帧间因子"：约束"从位姿A转到位姿B的相对运动"
#include <gtsam/nonlinear/NonlinearFactorGraph.h> // 因子图容器：收集所有因子
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h> // LM 优化求解器（一次性全局优化用）
#include <gtsam/nonlinear/Values.h>   // 存解的容器：每个节点(关键帧)一个位姿值
#include <gtsam/nonlinear/ISAM2.h>    // 增量式优化器（回环检测到后"快速重算受影响部分"）
#include <gtsam/linear/NoiseModel.h>  // 噪声模型：告诉优化器"某条约束我有多信任"

///// ========== 四、项目自定义头文件 ==========
#include "loop_closure.h"   // 回环检测模块（Scan Context 认路 + 配准）
#include "pose_pcd.hpp"     // "位姿+点云"的结构体（PosePcd），关键帧的数据格式
#include "utilities.hpp"    // 杂项工具函数（转换、保存等）

namespace fs = std::filesystem;            // 简写：fs::path 等
using namespace std::chrono;               // 用 chrono 的时间单位（毫秒等）方便计时

// 定义"里程计+点云"按近似时间对齐的同步策略，
// 意思是：把相差在允许范围内的一个 Odometry 消息和一个 PointCloud2 消息凑成一对，
// 之后它们会一起进入同一个回调（保证"这个点云对应的位姿就是这个里程计"）。
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> odom_pcd_sync_pol;

////////////////////////////////////////////////////////////////////////////////////////////////////
// =====================================================================
// 主类：FastLioSamScQn —— 整个 SLAM 系统的"总指挥部"
// 职责：接收(里程计+点云) → 维护关键帧 → 检测回环 → 优化因子图 → 输出修正轨迹/地图
// =====================================================================
class FastLioSamScQn
{
private:
    ///// ========== 1. 基本参数 ==========
    std::string map_frame_;      // 全局坐标系名（通常是 "map"），所有输出都转到这个系下
    std::string package_path_;   // 本 ROS 包的路径（用来定位配置/保存目录）
    std::string seq_name_;       // 本次数据集序列名（如 KITTI 的 "00"，用于 KITTI 格式保存）

    ///// ========== 2. 共享数据：里程计与点云（多线程读写，配了锁） ==========
    // 多个线程（里程计回调线程、回环检测线程、可视化线程）会同时访问下面这些，
    // 所以每个数据集都配一把互斥锁，避免数据竞争/崩溃。
    std::mutex realtime_pose_mutex_;    // 保护"实时位姿"相关数据
    std::mutex keyframes_mutex_;        // 保护关键帧序列 keyframes_
    std::mutex odom_callback_mutex_;    // 保护里程计回调里改动的共享变量
    std::mutex loop_timer_mutex_;       // 保护回环检测线程里的变量
    std::mutex loop_idx_pairs_mutex_;   // 保护 loop_idx_pairs_（回环索引对，供可视化读取）
    std::mutex graph_mutex_;            // 保护因子图 + ISAM2（后端优化线程专用）
    std::mutex vis_mutex_;              // 保护可视化数据（轨迹/地图被发布时加锁）

    Eigen::Matrix4d last_corrected_pose_ = Eigen::Matrix4d::Identity(); // 最近一次"回环修正后"的位姿（4x4 齐次矩阵），单位阵就表示"map 原点"
    Eigen::Matrix4d odom_delta_ = Eigen::Matrix4d::Identity();  //里程计增量 / 回环修正量：修改正后位姿与未修正位姿之差，用来把修正补偿给实时轨迹
    PosePcd current_frame_;              //current_frame_当前帧posePcd / 当前这一帧的（位姿+点云）
    std::vector<PosePcd> keyframes_;     //关键帧序列 / 从所有帧里挑出来的"值得记住的"帧集合
    int current_keyframe_idx_ = 0;       // 当前关键帧在 keyframes_ 里的下标（用于建图/回环索引）

    ///// ========== 3. 因子图与优化器（后端"账本"） ==========
    // 下面这些 bool 用 std::atomic 包起来，是因为它们会被"回环线程"写、被"可视化线程/优化线程"读，
    // atomic 保证跨线程读写不撕裂。
    std::atomic<bool> is_initialized_{false};     // 是否已初始化（处理过首帧），首帧要先建先验因子
    std::atomic<bool> loop_added_flag_{false};     // for opt：刚加入了一条回环约束 → 触发一次 ISAM2 优化
    std::atomic<bool> loop_added_flag_vis_{false}; // for vis：刚加入回环 → 触发一次可视化刷新
    std::shared_ptr<gtsam::ISAM2> isam_handler_ = nullptr;   //ISAM2优化 / GTSAM 的"增量式优化器"（账房先生本人）
    gtsam::NonlinearFactorGraph gtsam_graph_;                //因子图 / 账本：装所有节点(关键帧)和因子(约束)
    gtsam::Values init_esti_;          //优化位姿估计 / 优化之前每个关键帧的位姿初值（算前快照）
    gtsam::Values corrected_esti_;     // 优化之后每个关键帧的修正位姿（最终可信结果）
    double keyframe_thr_;              // 关键帧"平移阈值"（米）：两帧位移超过它 → 值得新增关键帧
    double keyframe_rotation_thr_rad_; // 关键帧"旋转阈值"（弧度）：两帧转角超过它 → 考虑新增（需配合下一项开关）
    bool enable_rotation_keyframes_ = true;   // 是否允许"仅因为转得够多"也新增关键帧（原地转圈也要记录）
    bool enable_adaptive_loop_noise_ = true;  // 回环因子噪声是否自适应：配准越准→噪声越小→优化器越信任这条回环
    bool enable_robust_loop_factor_ = true;   // 是否为回环因子加"鲁棒核"：防止某条错误回环把整张图带歪
    bool enable_quatro_ = true;               // 是否用 Quatro 做回环的全局粗配准（先不管初值使劲对，找大致相对位姿）
    bool enable_submap_matching_ = false;     // 是否用"子图"(附近好几个关键帧拼起来)去匹配回环（更稳但更慢）
    double voxel_res_;                        // 体素滤波器分辨率：配准/建图前先对点云降采样，提速+降噪
    fast_lio_sam_sc_qn::LoopFactorVarianceConfig loop_factor_variance_config_; // 回环因子噪声方差的配置（固定值/自适应系数等）
    int sub_key_num_;                         // 子图里包含几个关键帧（enable_submap_matching_ 为 true 时用）
    std::vector<std::pair<size_t, size_t>> loop_idx_pairs_; // for vis / 已检测到的回环"帧对"（如(3,50)表示第3帧和第50帧是同一地点），用于在 RViz 里画连线

    ///// ========== 4. 可视化 ==========
    tf::TransformBroadcaster broadcaster_;       // tf 广播器：把修正后的 map->odom 变换发布出去，供 RViz 使用
    pcl::PointCloud<pcl::PointXYZ> odoms_, corrected_odoms_; // 未修正/修正后的"关键帧位姿点"集合（在 RViz 里显示成轨迹点）
    nav_msgs::Path odom_path_, corrected_path_;  // 未修正/修正后的完整轨迹（Path 消息，RViz 里画成线）
    bool global_map_vis_switch_ = true;          // 是否把全局地图点云发布给 RViz 显示（帧数多时可关掉省性能）

    ///// ========== 5. 结果保存选项 ==========
    bool save_map_bag_ = false,          // 是否把地图存成 rosbag 文件
         save_map_pcd_ = false,          // 是否把地图存成 PCD 点云文件
         save_in_kitti_format_ = false;  // 是否额外保存 KITTI 格式的位姿文件（做评测用）

    ///// ========== 6. ROS 通信（发布器 / 订阅器 / 定时器） ==========
    ros::NodeHandle nh_;                 // 节点句柄：用它创建 pub/sub/timer/读参数
    // —— 修正后的结果（用户最关心的输出）——
    ros::Publisher corrected_odom_pub_,   // 发布"回环修正后"的里程计位姿
                     corrected_path_pub_, // 发布"修正后"的轨迹
                     odom_pub_,           // 转发原始(未修正)里程计
                     path_pub_;           // 转发原始(未修正)轨迹
    // —— 点云地图与回环显示 ——
    ros::Publisher corrected_current_pcd_pub_,  // 当前帧"修正后"转出来的点云
                     corrected_pcd_map_pub_,    // "修正后"的全局稠密点云地图
                     loop_detection_pub_;       // 回环检测的可视化 marker（连线/圆圈）
    ros::Publisher realtime_pose_pub_;          // 实时位姿(PoseStamped)，供导航/其他模块订阅
    // —— 回环配准的调试信息（看"认路认得好不好"）——
    ros::Publisher debug_src_pub_,              // 回环匹配的"源"点云（新来这帧）
                     debug_dst_pub_,            // 回环匹配的"目标"点云（历史帧）
                     debug_coarse_aligned_pub_, // Quatro 粗配准后的效果
                     debug_fine_aligned_pub_;   // 精配准(ICP)后的效果
    ros::Subscriber sub_save_flag_;      // 订阅"保存地图"指令（收到字符串消息即触发保存）
    ros::Timer loop_timer_,              // 周期定时器①：定时跑回环检测（独立线程）
               vis_timer_;               // 周期定时器②：定时刷新可视化（独立线程）
    // —— 里程计+点云的时间同步订阅 ——
    std::shared_ptr<message_filters::Synchronizer<odom_pcd_sync_pol>> sub_odom_pcd_sync_ = nullptr; // 同步器：把 odom 和 pcd 凑成时间对齐的一对
    std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> sub_odom_ = nullptr;           // 里程计订阅器（前端 FAST-LIO 发来）
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> sub_pcd_ = nullptr;      // 点云订阅器（与上一条同源）

    ///// ========== 7. 回环检测模块 ==========
    std::shared_ptr<LoopClosure> loop_closure_; // "认路员"：扫描历史关键帧找相似地点（Scan Context 描述子）+ 配准验证

public:
    // 构造函数：传入"私有参数句柄"（读 launch 里的参数）、创建订阅/同步器/定时器、初始化回环模块
    explicit FastLioSamScQn(const ros::NodeHandle &n_private);
    ~FastLioSamScQn();            // 析构：清理资源、保存遗留地图

private:
    ///// ========== 方法（内部工具函数） ==========
    // 根据某一帧的位姿，更新"未修正轨迹"和"修正轨迹"以及对应的轨迹点集合（供可视化）
    void updateOdomsAndPaths(const PosePcd &pose_pcd_in);
    // 判断这一帧是否值得作为"新的关键帧"：
    //   规则一般是 位移 > keyframe_thr_ 或 (转角 > keyframe_rotation_thr_rad_ 且 开了旋转关键帧开关)
    bool checkIfKeyframe(const PosePcd &pose_pcd_in, const PosePcd &latest_pose_pcd);
    // 根据"优化后的位姿"和 loop_idx_pairs_，生成回环的可视化 marker（在两端关键帧之间画线）
    visualization_msgs::Marker getLoopMarkers(const gtsam::Values &corrected_esti_in);

    ///// ========== 回调函数（事件驱动，哪个触发就跑哪段） ==========
    // ★ 核心回调：每当收到"时间对齐的(里程计, 点云)"一对就被唤醒。
    //   处理：更新当前帧 → 判断是否新关键帧 → 是则加入因子图 → 必要时触发优化/保存。
    void odomPcdCallback(const nav_msgs::OdometryConstPtr &odom_msg,
                         const sensor_msgs::PointCloud2ConstPtr &pcd_msg);
    // 收到"保存"命令字符串时触发：按 save_map_* 选项把地图存成 bag / PCD / KITTI
    void saveFlagCallback(const std_msgs::String::ConstPtr &msg);
    // ★ 周期执行（定时器 loop_timer_ 触发，独立线程）：
    //   调 loop_closure_ 检索回环 → Quatro+ICP 验证并算相对位姿 → 构造回环因子 → 置 loop_added_flag_ 触发 ISAM2 优化
    void loopTimerFunc(const ros::TimerEvent &event);
    // 周期执行（定时器 vis_timer_ 触发，独立线程）：
    //   刷新 tf、轨迹、地图点云、回环 marker 到 RViz
    void visTimerFunc(const ros::TimerEvent &event);
};

#endif