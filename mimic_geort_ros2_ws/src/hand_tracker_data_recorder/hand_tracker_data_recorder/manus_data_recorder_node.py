import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

HAND_TRACKER_RIGHT_NORMALIZED_PCLOUD_TOPIC_NAME = "/hand_tracker/right/normalized_pcloud_data"
HAND_TRACKER_LEFT_NORMALIZED_PCLOUD_TOPIC_NAME = "/hand_tracker/left/normalized_pcloud_data"

class ManusDataRecorderNode(Node):
    def __init__(self):
        super().__init__("manus_data_recorder")

        # Subscribers
        self.hand_tracker_right_normalized_pcloud_sub = self.create_subscription(
            Float32MultiArray,
            HAND_TRACKER_RIGHT_NORMALIZED_PCLOUD_TOPIC_NAME,
            self.hand_tracker_right_normalized_pcloud_callback,
            10,
        )

        self.hand_tracker_left_normalized_pcloud_sub = self.create_subscription(
            Float32MultiArray,
            HAND_TRACKER_LEFT_NORMALIZED_PCLOUD_TOPIC_NAME,
            self.hand_tracker_left_normalized_pcloud_callback,
            10,
        )

        # Example: buffers for streaming / logging
        self.right_buffer = []
        self.left_buffer = []

        # Optional timer to print stats periodically
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.get_logger().info("ManusDataRecorder node started.")

    def _msg_to_points(self, msg: Float32MultiArray) -> np.ndarray:
        """Convert Float32MultiArray message to Nx3 numpy array of points.
        
        Converts flat float array [x1, y1, z1, x2, y2, z2, ...] into shape (N, 3).
        """
        data = np.array(msg.data, dtype=np.float32)
        if data.size % 3 != 0:
            warning_msg = f"Received pointcloud with {data.size} elements, not divisible by 3."
            self.get_logger().warn(warning_msg)
            return data.reshape(-1)  # fallback

        points = data.reshape(-1, 3)
        return points

    def hand_tracker_right_normalized_pcloud_callback(self, msg: Float32MultiArray):
        points = self._msg_to_points(msg)
        self.right_buffer.append(points)
        self.get_logger().debug(f"Right hand frame: {points.shape}")

    def hand_tracker_left_normalized_pcloud_callback(self, msg: Float32MultiArray):
        points = self._msg_to_points(msg)
        self.left_buffer.append(points)
        self.get_logger().debug(f"Left hand frame: {points.shape}")

    def timer_callback(self):
        # Called every second — just an example
        right_len = len(self.right_buffer)
        left_len = len(self.left_buffer)
        self.get_logger().info(
            f"Recording MANUS data: right_frames={right_len}, left_frames={left_len}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ManusDataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ManusDataRecorderNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
