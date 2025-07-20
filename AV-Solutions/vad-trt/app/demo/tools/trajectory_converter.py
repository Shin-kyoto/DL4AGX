#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# QoS関連のクラスをインポート
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)

# メッセージ型をインポート
from autoware_planning_msgs.msg import Trajectory as InputTrajectory
from autoware_auto_planning_msgs.msg import Trajectory as OutputTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint as InputPoint
from autoware_auto_planning_msgs.msg import TrajectoryPoint as OutputPoint


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
            InputTrajectory,
            input_topic,
            self.listener_callback,
            qos_profile
        )
        
        self.get_logger().info(f"Converter started. Subscribing to '{input_topic}' and publishing to '{output_topic}'.")

    def listener_callback(self, msg: InputTrajectory):
        """
        メッセージ受信時のコールバック関数
        """
        output_msg = OutputTrajectory()
        output_msg.header = msg.header

        for input_point in msg.points:
            output_point = OutputPoint()
            
            output_point.time_from_start = input_point.time_from_start
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
