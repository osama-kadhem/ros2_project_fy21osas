import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav2_msgs.action import NavigateToPose
import signal
import math


class RobotProject(Node):

    def __init__(self):
        super().__init__('robot_project')

        self.bridge = CvBridge()
        self.sensitivity = 20

        # Current frame detection flags
        self.red_found   = False
        self.green_found = False
        self.blue_found  = False

        # Persistent flags
        self.red_seen   = False
        self.green_seen = False
        self.all_seen   = False

        # Blue tracking
        self.blue_cx     = None
        self.blue_area   = 0
        self.image_width = 640

        # Laser distance
        self.min_distance = float('inf')

        self.task_done = False

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)
        self.subscription

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.scan_sub

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.navigating = False
        self.waypoint_index = 0
        self.consecutive_rejections = 0

        # Waypoints covering the whole map
        self.waypoints = [
            ( 0.0,   0.0,  180),
            (-1.0,  -5.0,  270),
            ( 8.0, -12.0,  180),
            (-9.0, -14.0,   90),
            (-1.0,  -5.0,   90),
            (-10.0,  3.0,    0),
            ( 0.0,   0.0,  180),
        ]

        self.get_logger().info('RobotProject node started.')

    def scan_callback(self, msg):
        # Keep only forward facing readings
        ranges = msg.ranges
        n = len(ranges)
        front_ranges = ranges[:n//6] + ranges[-n//6:]
        valid = [r for r in front_ranges if msg.range_min < r < msg.range_max]
        self.min_distance = min(valid) if valid else float('inf')

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        h, w, _ = image.shape
        self.image_width = w
        s = self.sensitivity

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red mask
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,       100, 100]), np.array([s,       255, 255])),
            cv2.inRange(hsv, np.array([180 - s, 100, 100]), np.array([180,     255, 255]))
        )

        # Green mask
        green_mask = cv2.inRange(hsv,
            np.array([60 - s, 100, 100]),
            np.array([60 + s, 255, 255]))

        # Blue mask
        blue_mask = cv2.inRange(hsv,
            np.array([100, 150, 100]),
            np.array([140, 255, 255]))

        # Combined filtered image
        combined_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(green_mask, blue_mask))
        filtered_img  = cv2.bitwise_and(image, image, mask=combined_mask)

        # Detect and draw bounding boxes
        self.red_found,   image = self._detect_and_draw(image, red_mask,   (0, 0, 255), 'RED')
        self.green_found, image = self._detect_and_draw(image, green_mask, (0, 255, 0), 'GREEN')
        self.blue_found,  image = self._detect_and_draw(image, blue_mask,  (255, 0, 0), 'BLUE')

        # Update persistent seen flags
        if self.red_found and not self.red_seen:
            self.red_seen = True
            self.get_logger().info('RED seen for first time!')

        if self.green_found and not self.green_seen:
            self.green_seen = True
            self.get_logger().info('GREEN seen for first time!')

        if self.blue_found:
            self.get_logger().info(f'BLUE detected! Area={self.blue_area:.0f}, dist={self.min_distance:.2f}m')

        if self.red_seen and self.green_seen and not self.all_seen:
            self.all_seen = True
            self.get_logger().info('RED and GREEN both seen!')

        # Track blue contour
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if blue_contours:
            c = max(blue_contours, key=cv2.contourArea)
            self.blue_area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M['m00'] > 0:
                self.blue_cx = int(M['m10'] / M['m00'])
            else:
                self.blue_cx = None
        else:
            self.blue_area = 0
            self.blue_cx   = None

        # Show windows
        cv2.namedWindow('Camera Feed',    cv2.WINDOW_NORMAL)
        cv2.namedWindow('Filtered Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera Feed',    image)
        cv2.imshow('Filtered Image', filtered_img)
        cv2.resizeWindow('Camera Feed',    320, 240)
        cv2.resizeWindow('Filtered Image', 320, 240)
        cv2.waitKey(3)

    def _detect_and_draw(self, image, mask, colour_bgr, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:
                found = True
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + bw, y + bh), colour_bgr, 2)
                cv2.putText(image, label, (x, max(y - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour_bgr, 2)
        return found, image

    def send_next_waypoint(self):
        if self.waypoint_index >= len(self.waypoints):
            self.waypoint_index = 0

        x, y, yaw_deg = self.waypoints[self.waypoint_index]
        self.waypoint_index += 1
        self.consecutive_rejections = 0

        self.get_logger().info(f'Navigating to waypoint ({x:.1f}, {y:.1f})')

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        yaw = math.radians(yaw_deg)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal_msg.pose = pose
        self.navigating = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Waypoint rejected, trying next.')
            self.navigating = False
            self.consecutive_rejections += 1
            if self.consecutive_rejections > 3:
                self.waypoint_index += 1
                self.consecutive_rejections = 0
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        self.get_logger().info('Waypoint reached.')
        self.navigating = False

    def hard_stop(self):
        # Publish zero velocity many times to ensure robot fully stops
        stop_msg = Twist()
        stop_msg.linear.x  = 0.0
        stop_msg.linear.y  = 0.0
        stop_msg.linear.z  = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0
        for _ in range(20):
            self.publisher.publish(stop_msg)
        self.get_logger().info('Hard stop executed.')

    def approach_blue(self):
        STOP_DISTANCE = 1.0
        twist = Twist()

        self.get_logger().info(
            f'Approaching blue, area={self.blue_area:.0f}, dist={self.min_distance:.2f}m')

        # Stop using laser distance
        if self.min_distance <= STOP_DISTANCE and self.blue_found:
            self.get_logger().info(
                f'TASK COMPLETE - stopped at {self.min_distance:.2f}m from blue box!')
            self.hard_stop()
            self.task_done = True
            return

        if self.blue_cx is not None:
            error = self.blue_cx - self.image_width / 2.0
            twist.angular.z = -error / 300.0
            twist.linear.x  = 0.15
        else:
            twist.angular.z = 0.3
            twist.linear.x  = 0.0

        self.publisher.publish(twist)

    def stop(self):
        self.publisher.publish(Twist())


def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = RobotProject()

    signal.signal(signal.SIGINT, signal_handler)

    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    robot.get_logger().info('Waiting for Nav2 action server...')
    robot.nav_client.wait_for_server()
    robot.get_logger().info('Nav2 ready, starting exploration.')

    robot.send_next_waypoint()

    try:
        while rclpy.ok():
            if robot.task_done:
                break

            if robot.blue_found and robot.blue_area > 500:
                robot.approach_blue()

            elif robot.green_seen and not robot.red_seen and not robot.navigating:
                twist = Twist()
                twist.angular.z = 0.5
                robot.publisher.publish(twist)

            elif not robot.navigating:
                robot.send_next_waypoint()

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    robot.hard_stop()


if __name__ == '__main__':
    main()