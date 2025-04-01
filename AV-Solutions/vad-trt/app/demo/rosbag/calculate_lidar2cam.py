#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import rosbag2_py
from rclpy.serialization import serialize_message, deserialize_message
import argparse
import os
import sys
import time
import numpy as np
from rosidl_runtime_py.utilities import get_message
import copy
from tqdm import tqdm
from pyquaternion import Quaternion

# tf2_ros 関連のインポート
import tf2_ros
from tf2_ros import Buffer, TransformListener


def main():
    parser = argparse.ArgumentParser(description='Process rosbag: copy all topics and add camera transforms to tf_static')
    parser.add_argument('--rosbag', default="/home/autoware/rosbags/simpl_vs_autoware/complete_bag/", help='Path to the input rosbag')
    args = parser.parse_args()
    
    # rosbagを開き，中に入っているtf_staticの情報から，camera0/camera_optical_link to /base_linkのtf_staticを取り出す
    try:
        # ROSバッグの設定
        storage_options = rosbag2_py._storage.StorageOptions(
            uri=args.rosbag,
            storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        
        # リーダーの初期化
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # 目的のカメラフレーム
        target_frame = 'base_link'
        parent_frame = 'camera0/camera_optical_link'
        transform_found = False
        
        print(f"{target_frame}から{parent_frame}へのtf_staticを検索中...")
        
        # ROSバッグからメッセージを読み込む
        while reader.has_next() and not transform_found:
            (topic, data, timestamp) = reader.read_next()
            
            # tf_staticトピックを処理
            if topic == '/tf_static':
                # TFMessageをデシリアライズ
                msg = TFMessage()
                msg = deserialize_message(data, get_message('tf2_msgs/msg/TFMessage'))
                
                # 各変換を確認
                for transform in msg.transforms:
                    if transform.child_frame_id == target_frame and transform.header.frame_id == parent_frame:
                        # 目的の変換が見つかった
                        print(f"変換が見つかりました: parent: {parent_frame} -> target: {target_frame}")
                        print(f"平行移動: [{transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}]")
                        print(f"回転: [{transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}]")
                        # pyquaternionを使用して回転行列に変換
                        q = Quaternion(w=transform.transform.rotation.w,
                                     x=transform.transform.rotation.x,
                                     y=transform.transform.rotation.y,
                                     z=transform.transform.rotation.z)
                        rotation_sensor2aw = q.rotation_matrix
                        print("回転行列:")
                        print(rotation_sensor2aw)

                        # 平行移動をnumpy配列に変換
                        translation_aw2sensor = np.array([
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z
                        ])

                        def get_lidar2cam(translation_aw2sensor, rotation_sensor2aw):
                            # -1 translation * rotation_matrix を計算
                            translation_sensor2aw = -1 * np.dot(translation_aw2sensor, rotation_sensor2aw)
                            print("translation_sensor2aw:")
                            print(translation_sensor2aw)

                            # rotation_aw2nsを用いた変換
                            rotation_aw2ns = np.array([[0.0, 1.0, 0.0],[-1.0, 0.0, 0.0],[0.0, 0.0, 1.0],])
                            translation_sensor2ns = np.dot(translation_sensor2aw, rotation_aw2ns)
                            print("translation_sensor2ns:")
                            print(translation_sensor2ns)

                            # rotation_ns2sensorの計算
                            rotation_ns2sensor = np.dot(rotation_sensor2aw, rotation_aw2ns).T
                            print("rotation_ns2sensor:")
                            print(rotation_ns2sensor)

                            # translation_ns2sensorの計算
                            translation_ns2sensor = -np.dot(rotation_ns2sensor, translation_sensor2ns) # なぜか，-1がないと合う
                            print("translation_ns2sensor:")
                            print(translation_ns2sensor)

                            lidar2cam_rt = np.eye(4)
                            lidar2cam_rt[:3,:3] = rotation_ns2sensor.T
                            lidar2cam_rt[3, :3] = translation_ns2sensor

                            return lidar2cam_rt
                        
                        lidar2cam_rt = get_lidar2cam(translation_aw2sensor=translation_aw2sensor, rotation_sensor2aw=rotation_sensor2aw)
                        print("lidar2cam_rt.T:")
                        print(lidar2cam_rt.T)

                        # import pdb;pdb.set_trace()
                        
                        transform_found = True
                        break

        
        if not transform_found:
            print(f"警告: {target_frame}と{parent_frame}間の変換が見つかりませんでした。")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
