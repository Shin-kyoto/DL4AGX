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
from rosidl_runtime_py.utilities import get_message
import copy
from tqdm import tqdm

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
                        transform_found = True
                        break
                    elif transform.header.frame_id == target_frame and transform.child_frame_id == parent_frame:
                        # 逆方向の変換が見つかった場合
                        print(f"逆方向の変換が見つかりました: {target_frame} -> {parent_frame}")
                        print(f"平行移動: [{transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}]")
                        print(f"回転: [{transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}]")
                        transform_found = True
                        break
        
        if not transform_found:
            print(f"警告: {target_frame}と{parent_frame}間の変換が見つかりませんでした。")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
