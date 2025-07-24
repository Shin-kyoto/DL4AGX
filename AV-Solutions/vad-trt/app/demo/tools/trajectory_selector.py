#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
# QoS関連のクラスをインポート
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)

# メッセージ型をインポート
from autoware_internal_planning_msgs.msg import CandidateTrajectories
from autoware_internal_planning_msgs.msg import CandidateTrajectory as InputTrajectory
from autoware_auto_planning_msgs.msg import Trajectory as OutputTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint as InputPoint
from autoware_auto_planning_msgs.msg import TrajectoryPoint as OutputPoint
from nav_msgs.msg import Odometry


class TrajectoryConverter(Node):
    """
    Trajectoryメッセージをある型から別の型へ変換し、再配信するノード。
    指定されたQoSプロファイルを使用する。
    """
    def __init__(self):
        super().__init__('trajectory_converter_node')

        self.declare_parameter('input_topic', '/input/trajectory')
        self.declare_parameter('output_topic', '/output/trajectory')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # --- 指定されたQoSプロファイルの定義 ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE  # VOLATILEに設定
        )
        # ------------------------------------

        # PublisherとSubscriberの作成時にQoSプロファイルを渡す
        self.publisher_ = self.create_publisher(
            OutputTrajectory,
            output_topic,
            qos_profile
        )
        self.subscription = self.create_subscription(
            CandidateTrajectories,
            input_topic,
            self.listener_callback,
            qos_profile
        )
        
        # Step1: /planning/scenario_planning/trajectoryと/localization/kinematic_stateを取得するためのsubscriber
        # trajectory用のQoS設定（PublisherがBEST_EFFORTなので合わせる）
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # kinematic_state用のQoS設定（PublisherがRELIABLEなので合わせる）
        kinematic_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # trajectory subscriber
        self.trajectory_sub = self.create_subscription(
            OutputTrajectory,
            '/planning/scenario_planning/trajectory',
            self.trajectory_callback,
            trajectory_qos
        )
        
        # kinematic state subscriber
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.odometry_callback,
            kinematic_qos
        )
        
        # データを格納する変数
        self.current_trajectory = None
        self.current_odometry = None
        
        # 連続選択を追跡する変数
        self.previous_selected_index = None
        self.consecutive_count = 0
        
        # 10.0倍適用の残り回数を追跡する変数
        self.high_multiplier_remaining = 0
        
        self.get_logger().info(f"Converter started. Subscribing to '{input_topic}' and publishing to '{output_topic}'.")

    def trajectory_callback(self, msg: OutputTrajectory):
        """trajectory受信時のコールバック"""
        self.current_trajectory = msg
        self.get_logger().debug("Received trajectory")

    def odometry_callback(self, msg: Odometry):
        """odometry受信時のコールバック"""
        self.current_odometry = msg
        self.get_logger().debug("Received odometry")

    def find_nearest_index(self, trajectory_points, position):
        """
        simple_pure_pursuit.cppのfindNearestIndex相当の処理
        trajectory_pointsから現在位置に最も近いポイントのインデックスを見つける
        """
        if not trajectory_points:
            return 0
            
        min_distance = float('inf')
        nearest_index = 0
        
        for i, point in enumerate(trajectory_points):
            dx = point.pose.position.x - position.x
            dy = point.pose.position.y - position.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
                
        return nearest_index

    def get_transform_matrix(self, odometry_msg):
        """
        vad_node.cppのget_transform_matrix相当の処理
        odometryメッセージからmap2base変換行列を計算
        """
        # 位置を取得
        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y
        z = odometry_msg.pose.pose.position.z
        
        # クォータニオンを取得
        qx = odometry_msg.pose.pose.orientation.x
        qy = odometry_msg.pose.pose.orientation.y
        qz = odometry_msg.pose.pose.orientation.z
        qw = odometry_msg.pose.pose.orientation.w
        
        # クォータニオンを正規化
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-8:
            # ゼロクォータニオンの場合は単位クォータニオンを使用
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # クォータニオンから回転行列を計算
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # 並進ベクトル
        t = np.array([x, y, z])
        
        # Base_link → Map (forward transform)
        base2map = np.eye(4)
        base2map[:3, :3] = R
        base2map[:3, 3] = t
        
        # Map → Base_link (inverse transform)
        map2base = np.eye(4)
        map2base[:3, :3] = R.T
        map2base[:3, 3] = -R.T @ t
        
        return base2map, map2base

    def listener_callback(self, msg: CandidateTrajectories):
        """
        メッセージ受信時のコールバック関数
        Step1-3を統合した処理
        """
        # Step1: データの完整性チェック
        if self.current_trajectory is None or self.current_odometry is None:
            self.get_logger().warn("Trajectory or odometry data not available yet")
            return
        
        if not self.current_trajectory.points:
            self.get_logger().warn("Trajectory points is empty")
            return
        
        if not msg.candidate_trajectories:
            self.get_logger().warn("No candidate trajectories available")
            return
        
        # Step1: closest_pointを取得
        closest_traj_point_idx = self.find_nearest_index(
            self.current_trajectory.points, 
            self.current_odometry.pose.pose.position
        )
        
        # Step1: closest_traj_point_idxから5つ先まで先読み
        lookahead_count = 10
        end_idx = min(closest_traj_point_idx + lookahead_count, len(self.current_trajectory.points))
        lookahead_points = self.current_trajectory.points[closest_traj_point_idx:end_idx]
        
        if len(lookahead_points) == 0:
            self.get_logger().warn("No lookahead points available")
            return
        
        # Step2: 5つの点をbase_link座標系へ変換
        base2map, map2base = self.get_transform_matrix(self.current_odometry)
        
        y_base_values = []
        for point in lookahead_points:
            # 各点のmap座標系での位置
            point_xyz_map = np.array([
                point.pose.position.x,
                point.pose.position.y,
                point.pose.position.z,
                1.0  # 同次座標
            ])
            
            # base_link座標系に変換
            point_xyz_base = map2base @ point_xyz_map
            y_base_values.append(point_xyz_base[1])  # y座標を記録
        
        # Step3: 変換した点を使って正負判定
        # 10個の点のプラスとマイナスの個数に基づいて判定
        positive_count = sum(1 for y in y_base_values if y > 0.1)
        negative_count = sum(1 for y in y_base_values if y < -0.1)
        total_count = len(y_base_values)
        
        # 後半5個を取得（利用可能な範囲で）
        latter_half_start = max(0, total_count - 5)
        latter_half_values = y_base_values[latter_half_start:]
        latter_half_negative_count = sum(1 for y in latter_half_values if y < -0.1)
        
        # 判定ロジック：
        # 過半数がマイナスかつ、後半5個がすべてマイナスなら0
        majority_negative = negative_count > 7
        latter_half_all_negative = latter_half_negative_count == len(latter_half_values)
        
        if majority_negative and latter_half_all_negative:
            selected_index = 0
            self.get_logger().info(f"Selected trajectory 0 (majority negative: {negative_count}/{total_count}, latter half all negative: {latter_half_negative_count}/{len(latter_half_values)})")
            self.get_logger().info(f"Y values: {[f'{y:.2f}' for y in y_base_values]}")
        elif positive_count > negative_count:
            selected_index = 1  
            self.get_logger().info(f"Selected trajectory 1 (more positive: {positive_count} vs {negative_count})")
            self.get_logger().info(f"Y values: {[f'{y:.2f}' for y in y_base_values]}")
        else:
            selected_index = 2
            self.get_logger().info(f"Selected trajectory 2 (default case: pos={positive_count}, neg={negative_count})")
            self.get_logger().info(f"Y values: {[f'{y:.2f}' for y in y_base_values]}")
        
        # 連続選択カウントを更新
        if self.previous_selected_index == selected_index:
            self.consecutive_count += 1
            # 0番が8回連続で選ばれた時、10.0倍適用の8回をセット
            if selected_index == 0 and self.consecutive_count == 8:
                self.high_multiplier_remaining = 8
                self.get_logger().info("0番が8回連続選択されました。次の8回は10.0倍を適用します。")
        else:
            self.consecutive_count = 1
            # 異なるtrajectoryが選ばれた場合、10.0倍適用をリセット
            if self.high_multiplier_remaining > 0:
                self.get_logger().info("異なる軌道が選択されたため、10.0倍適用をリセットします。")
                self.high_multiplier_remaining = 0
        self.previous_selected_index = selected_index
        
        # 選択されたtrajectoryを確認
        if selected_index >= len(msg.candidate_trajectories):
            self.get_logger().error(f"Selected index {selected_index} is out of range (available: {len(msg.candidate_trajectories)})")
            return
        
        # 選択されたcandidate trajectoryを変換して出力
        input_msg = msg.candidate_trajectories[selected_index]
        output_msg = OutputTrajectory()
        output_msg.header = input_msg.header

        for input_point in input_msg.points:
            output_point = OutputPoint()
            
            output_point.time_from_start = input_point.time_from_start
            
            if selected_index == 0:
                # poseをコピーしてy座標を変更
                output_point.pose = input_point.pose
                # 13.0倍適用の残り回数があれば13.0倍、そうでなければ5.0倍
                if self.high_multiplier_remaining > 0:
                    y_multiplier = 13.0
                    # 最初のポイント処理時のみ残り回数を減らす
                    if input_point == input_msg.points[0]:
                        self.high_multiplier_remaining -= 1
                        self.get_logger().info(f"13.0倍適用中（残り回数: {self.high_multiplier_remaining}）")
                else:
                    y_multiplier = 4.0

                output_point.pose.position.y = input_point.pose.position.y * y_multiplier
                
                # 初回のログ出力時のみ倍率を表示
                if input_point == input_msg.points[0]:  # 最初のポイントの時のみログ出力
                    self.get_logger().info(f"Applying y-coordinate multiplier: {y_multiplier} (consecutive count: {self.consecutive_count})")
            else:
                output_point.pose = input_point.pose
            
            output_point.longitudinal_velocity_mps = input_point.longitudinal_velocity_mps
            output_point.lateral_velocity_mps = input_point.lateral_velocity_mps
            output_point.acceleration_mps2 = input_point.acceleration_mps2
            output_point.heading_rate_rps = input_point.heading_rate_rps
            output_point.front_wheel_angle_rad = input_point.front_wheel_angle_rad
            output_point.rear_wheel_angle_rad = input_point.rear_wheel_angle_rad

            output_msg.points.append(output_point)

        self.publisher_.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
