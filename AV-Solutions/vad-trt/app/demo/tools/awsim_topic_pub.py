#!/usr/bin/env python3

import argparse
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS2 message imports
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from std_msgs.msg import Header


class AwsimTopicPublisher(Node):
    def __init__(self, config_path):
        super().__init__('awsim_topic_publisher')
        
        # Load configuration from YAML
        self.config = self.load_config(config_path)
        self.bridge = CvBridge()
        
        # Setup QoS profiles based on C++ implementation
        self.sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.camera_info_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.reliable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.tf_static_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Initialize publishers and subscribers
        self.setup_publishers()
        self.setup_subscribers()
        
        # Store original TF static message
        self.tf_static_original = None
        
        self.get_logger().info('AWSIM Topic Publisher initialized')

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.get_logger().info(f'Loaded configuration from {config_path}')
                return config
        except Exception as e:
            self.get_logger().error(f'Failed to load config: {e}')
            raise

    def setup_publishers(self):
        """Setup publishers for camera topics, camera_info topics, and tf"""
        self.camera_publishers = {}
        self.camera_info_publishers = {}
        
        # Create publishers for each camera topic
        for topic_name in self.config['publish']['camera_topics']:
            self.camera_publishers[topic_name] = self.create_publisher(
                CompressedImage, topic_name, self.sensor_qos
            )
            
        # Create publishers for each camera_info topic
        for topic_name in self.config['publish']['camera_info_topics']:
            self.camera_info_publishers[topic_name] = self.create_publisher(
                CameraInfo, topic_name, self.camera_info_qos
            )
            
        # Create TF static publisher
        self.tf_static_publisher = self.create_publisher(
            TFMessage, '/tf_static', self.tf_static_qos
        )

    def setup_subscribers(self):
        """Setup subscribers for camera topic and tf"""
        # Subscribe to camera topic
        self.camera_subscriber = self.create_subscription(
            Image,
            self.config['subscribe']['camera_topic'],
            self.camera_callback,
            self.sensor_qos
        )
        
        # Subscribe to TF static
        self.tf_static_subscriber = self.create_subscription(
            TFMessage,
            self.config['subscribe']['tf_topic'],
            self.tf_static_callback,
            self.tf_static_qos
        )

    def camera_callback(self, msg):
        """Callback for camera topic subscription"""
        # Get current timestamp
        current_time = self.get_clock().now().to_msg()
        
        # Publish camera topics
        self.publish_camera_topics(current_time)
        
        # Publish camera info topics
        self.publish_camera_info_topics(current_time)
        
        # Publish updated TF static if available
        if self.tf_static_original is not None:
            self.publish_tf_static(current_time)

    def tf_static_callback(self, msg):
        """Callback for TF static subscription"""
        self.tf_static_original = msg
        self.get_logger().info('Received TF static message')

    def create_camera_topic(self, topic_name, width, height, stamp):
        """Create a black camera image"""
        # Create black image
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Compress image to JPEG
        _, buffer = cv2.imencode('.jpg', black_image)
        
        # Create CompressedImage message
        compressed_image_msg = CompressedImage()
        compressed_image_msg.header.stamp = stamp
        compressed_image_msg.header.frame_id = 'camera_optical_link'
        compressed_image_msg.format = 'jpeg'
        compressed_image_msg.data = buffer.tobytes()
        
        return compressed_image_msg

    def create_camera_info_topic(self, topic_name, k_matrix, stamp):
        """Create camera_info topic"""
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = stamp
        camera_info_msg.header.frame_id = 'camera_optical_link'
        
        # Set image dimensions
        camera_info_msg.width = self.config['camera']['image_size']['width']
        camera_info_msg.height = self.config['camera']['image_size']['height']
        
        # Set camera matrix K
        camera_info_msg.k = k_matrix
        
        # Set distortion model (assuming no distortion for simplicity)
        camera_info_msg.distortion_model = 'plumb_bob'
        camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Set rectification matrix (identity for no rectification)
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # Set projection matrix (same as K for no rectification)
        camera_info_msg.p = [
            k_matrix[0], k_matrix[1], k_matrix[2], 0.0,
            k_matrix[3], k_matrix[4], k_matrix[5], 0.0,
            k_matrix[6], k_matrix[7], k_matrix[8], 0.0
        ]
        
        return camera_info_msg

    def update_tf_static(self, tf_static_original, camera_links_dict, stamp):
        """Update tf_static_original with extrinsic information"""
        updated_tf_msg = TFMessage()
        
        # Copy original transforms
        updated_tf_msg.transforms = list(tf_static_original.transforms)
        
        # Add or update camera optical link transforms
        for link_name, transform_values in camera_links_dict.items():
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = 'base_link'  # Assuming base_link as parent
            transform.child_frame_id = link_name
            
            # Set translation
            transform.transform.translation.x = transform_values.get('x', 0.0)
            transform.transform.translation.y = transform_values.get('y', 0.0)
            transform.transform.translation.z = transform_values.get('z', 0.0)
            
            # Set rotation (quaternion)
            transform.transform.rotation.x = transform_values.get('qx', 0.0)
            transform.transform.rotation.y = transform_values.get('qy', 0.0)
            transform.transform.rotation.z = transform_values.get('qz', 0.0)
            transform.transform.rotation.w = transform_values.get('qw', 1.0)
            
            # Check if this transform already exists and update it, or add new one
            existing_transform_found = False
            for i, existing_transform in enumerate(updated_tf_msg.transforms):
                if existing_transform.child_frame_id == link_name:
                    updated_tf_msg.transforms[i] = transform
                    existing_transform_found = True
                    break
            
            if not existing_transform_found:
                updated_tf_msg.transforms.append(transform)
        
        return updated_tf_msg

    def publish_camera_topics(self, stamp):
        """Publish camera topics"""
        width = self.config['camera']['image_size']['width']
        height = self.config['camera']['image_size']['height']
        
        for topic_name in self.config['publish']['camera_topics']:
            image_msg = self.create_camera_topic(topic_name, width, height, stamp)
            self.camera_publishers[topic_name].publish(image_msg)

    def publish_camera_info_topics(self, stamp):
        """Publish camera_info topics"""
        k_matrix = self.config['camera']['k_matrix']
        
        for topic_name in self.config['publish']['camera_info_topics']:
            camera_info_msg = self.create_camera_info_topic(topic_name, k_matrix, stamp)
            self.camera_info_publishers[topic_name].publish(camera_info_msg)

    def publish_tf_static(self, stamp):
        """Publish updated TF static"""
        camera_links = self.config['tf']['camera_optical_links']
        updated_tf_msg = self.update_tf_static(
            self.tf_static_original, camera_links, stamp
        )
        self.tf_static_publisher.publish(updated_tf_msg)


def main():
    parser = argparse.ArgumentParser(description='AWSIM Topic Publisher')
    parser.add_argument('config_path', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = AwsimTopicPublisher(args.config_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()