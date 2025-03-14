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

class RosbagProcessor(Node):
    def __init__(self, rosbag_path, use_pdb=False):
        super().__init__('rosbag_processor')
        
        self.use_pdb = use_pdb
        self.rosbag_path = rosbag_path
        
        # フレームをハードコード
        self.parent_frame = 'base_link'
        self.child_frames = [
            'camera0/camera_optical_link',
            'camera1/camera_optical_link',
            'camera2/camera_optical_link',
            'camera3/camera_optical_link',
            'camera4/camera_optical_link',
            'camera5/camera_optical_link'
        ]
        
        # tf_static の間隔と数を設定
        self.tf_static_interval_ms = 100  # 100ms間隔
        self.tf_static_count = 600  # 600個のメッセージ
        
        # tf2 バッファとリスナーの初期化
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 変換を保存する辞書
        self.transforms = {}
        self.transforms_complete = False
        
        # タイマーを設定して、tf2からの変換を取得する時間を与える
        self.create_timer(1.0, self.check_transforms)
        
        # タイムスタンプを保持
        self.first_timestamp = None
        
        # tf_staticメッセージを保持
        self.tf_static_msg = None
        self.tf_static_received = False
        
        # Subscribe to tf_static with transient_local durability
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf_static',
            self.tf_static_callback,
            qos_profile
        )
        
        self.get_logger().info('Subscribed to /tf_static')
    
    def check_transforms(self):
        """すべてのカメラフレームの変換を取得する"""
        if self.transforms_complete:
            return
            
        all_found = True
        for child_frame in self.child_frames:
            if child_frame in self.transforms:
                continue
                
            try:
                # lookupTransform を呼び出して、指定したフレーム間の変換を取得
                transform = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    child_frame,
                    rclpy.time.Time()
                )
                
                self.get_logger().info(f'Found transform from {self.parent_frame} to {child_frame}')
                self.get_logger().info(f'Translation: [{transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}]')
                self.get_logger().info(f'Rotation: [{transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}]')
                
                self.transforms[child_frame] = transform
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f'Could not find transform to {child_frame}: {e}')
                all_found = False
        
        # すべての変換が見つかったか、最大試行回数に達したかをチェック
        found_count = len(self.transforms)
        total_count = len(self.child_frames)
        self.get_logger().info(f'Found {found_count} of {total_count} transforms')
        
        if found_count > 0 and (all_found or found_count == total_count):
            self.transforms_complete = True
            self.get_logger().info('All available transforms found')
            
            # /tf_staticが受信済みでtransformsが見つかったら、処理を開始
            if self.tf_static_received:
                self.process_rosbag()
    
    def tf_static_callback(self, msg):
        """tf_staticメッセージを受信したら保存"""
        if not self.tf_static_received:
            self.get_logger().info(f'Received /tf_static message with {len(msg.transforms)} transforms')
            self.tf_static_msg = msg
            self.tf_static_received = True
            
            # transformsが見つかっていたら、処理を開始
            if self.transforms_complete:
                self.process_rosbag()
    
    def process_rosbag(self):
        """rosbagの処理を開始"""
        self.get_logger().info('Starting rosbag processing')
        
        # 入力rosbagの読み込み準備
        reader = rosbag2_py.SequentialReader()
        storage_options_in = rosbag2_py._storage.StorageOptions(
            uri=self.rosbag_path,
            storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader.open(storage_options_in, converter_options)
        
        # トピック情報の取得
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_info.name: topic_info.type for topic_info in topic_types}
        
        # 最初のタイムスタンプの取得
        self.first_timestamp = self.get_first_timestamp(reader)
        self.get_logger().info(f'First timestamp: {self.first_timestamp}')
        
        # 出力rosbagの準備
        output_dir = os.path.join(os.path.dirname(self.rosbag_path), 'complete_bag')
        
        writer = rosbag2_py.SequentialWriter()
        storage_options_out = rosbag2_py._storage.StorageOptions(
            uri=output_dir,
            storage_id='sqlite3')
        writer.open(storage_options_out, converter_options)
        
        # 入力rosbagのすべてのトピックを出力rosbagに作成
        created_topics = set()
        for topic_info in topic_types:
            writer.create_topic(topic_info)
            created_topics.add(topic_info.name)
            self.get_logger().info(f'Created topic: {topic_info.name}')
        
        # tf_staticトピックがまだ作成されていなければ作成
        if '/tf_static' not in created_topics:
            tf_static_topic = rosbag2_py._storage.TopicMetadata(
                name='/tf_static',
                type='tf2_msgs/msg/TFMessage',
                serialization_format='cdr',
                offered_qos_profiles='- history: 1\n  depth: 1\n  reliability: 1\n  durability: 3\n  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n  avoid_ros_namespace_conventions: false'
            )
            writer.create_topic(tf_static_topic)
            self.get_logger().info('Created /tf_static topic')
        
        # Debug with pdb if use_pdb is True
        if self.use_pdb:
            import pdb
            self.get_logger().info('Starting pdb debugger for tf_static inspection')
            self.get_logger().info('Tip: Use "p self.tf_static_msg" to print the received tf_static message')
            self.get_logger().info('Tip: Use "p self.transforms" to see all transforms from tf2')
            self.get_logger().info('Tip: Use "c" to continue execution')
            pdb.set_trace()
        
        # tf_staticメッセージの処理
        if self.tf_static_msg:
            # 取得した変換をtf_staticに追加
            for child_frame, transform in self.transforms.items():
                # タイムスタンプを最初のタイムスタンプに更新
                transform.header.stamp.sec = int(self.first_timestamp / 1000000000)
                transform.header.stamp.nanosec = int(self.first_timestamp % 1000000000)
                
                # 変換をメッセージに追加
                self.tf_static_msg.transforms.append(transform)
                self.get_logger().info(f'Added transform from {self.parent_frame} to {child_frame}')
            
            self.get_logger().info(f'Updated /tf_static message has {len(self.tf_static_msg.transforms)} transforms')
            
            # 100ms間隔で600個のtf_staticメッセージを作成
            self.get_logger().info(f'Writing {self.tf_static_count} tf_static messages at {self.tf_static_interval_ms}ms intervals')
            
            from tqdm import tqdm
            
            current_timestamp = self.first_timestamp
            interval_ns = self.tf_static_interval_ms * 1000000  # 100msをナノ秒に変換
            
            # tqdmを使って進捗バーを表示
            for i in tqdm(range(self.tf_static_count), desc="Writing tf_static messages", unit="msg"):
                # 各メッセージに異なるタイムスタンプを設定
                current_msg = copy.deepcopy(self.tf_static_msg)
                
                # すべての変換のタイムスタンプを更新
                for transform in current_msg.transforms:
                    transform.header.stamp.sec = int(current_timestamp / 1000000000)
                    transform.header.stamp.nanosec = int(current_timestamp % 1000000000)
                
                # メッセージをシリアライズして書き込み
                serialized_msg = serialize_message(current_msg)
                writer.write('/tf_static', serialized_msg, current_timestamp)
                
                # タイムスタンプを更新
                current_timestamp += interval_ns
            
            self.get_logger().info(f'Wrote {self.tf_static_count} tf_static messages to output rosbag')
        
        # 読み取り位置をリセット
        reader.reset_filter()
        
        # 残りのメッセージをコピー（/tf_staticはスキップ）
        copy_count = 0
        
        # まずはrosbagのサイズから概算のメッセージ数を取得する（正確ではないが目安として使用）
        # これは実際には使わず、動的に更新されるtqdmを使用
        self.get_logger().info(f'Copying messages from input rosbag...')
        
        # tqdmを使って進捗バーを表示（初期値は適当に設定し、後で更新）
        from tqdm import tqdm
        pbar = tqdm(desc="Copying messages", unit="msg")
        
        # ステータス更新用のカウンタ
        last_update_time = time.time()
        update_interval = 0.5  # 秒単位でのアップデート間隔
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            # tf_staticメッセージはすでに書き込み済みのためスキップ
            if topic == '/tf_static':
                continue
            
            # メッセージを書き込み
            writer.write(topic, data, timestamp)
            copy_count += 1
            
            # 必要に応じてプログレスバーを更新（頻繁な更新を避けるため時間ベースでチェック）
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                pbar.n = copy_count
                pbar.refresh()
                last_update_time = current_time
        
        # 最終更新
        pbar.n = copy_count
        pbar.refresh()
        pbar.close()
        
        self.get_logger().info(f'Copied a total of {copy_count} messages')
        # writer.close() は使えないため、代わりに del writer を使用して暗黙的に解放する
        del writer
        self.get_logger().info('Rosbag processing completed')
        
        # 処理が完了したらシャットダウン
        rclpy.shutdown()
    
    def get_first_timestamp(self, reader):
        """rosbagの最初のタイムスタンプを取得"""
        reader.reset_filter()
        
        # 最初のメッセージのタイムスタンプを取得
        first_timestamp = None
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            first_timestamp = timestamp
            break
        
        # タイムスタンプが見つからなかった場合はエラー
        if first_timestamp is None:
            self.get_logger().error('Could not read any message from the rosbag')
            sys.exit(1)
        
        # 読み取り位置をリセット
        reader.reset_filter()
        return first_timestamp

def main():
    parser = argparse.ArgumentParser(description='Process rosbag: copy all topics and add camera transforms to tf_static')
    parser.add_argument('--rosbag', default="/home/autoware/rosbags/simpl_vs_autoware/6c9de6ec-1328-4fbc-93a1-51407dcdf81e_2", help='Path to the input rosbag')
    parser.add_argument('--use_pdb', action='store_true', help='Enable pdb debugger for inspecting tf_static message')
    parser.add_argument('--interval', type=int, default=100, help='Interval between tf_static messages in milliseconds')
    parser.add_argument('--count', type=int, default=600, help='Number of tf_static messages to generate')
    args = parser.parse_args()
    
    rclpy.init()
    
    processor = RosbagProcessor(args.rosbag, use_pdb=args.use_pdb)
    processor.tf_static_interval_ms = args.interval
    processor.tf_static_count = args.count
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure everything is properly closed
        processor.destroy_node()
        if not rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
