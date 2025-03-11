import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
import cv2


def bin_to_pointcloud2(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, 7)
        # data = data[:, [0, 1, 2, 3, 5]]  # Keep only the first 4 columns and the 6th column of data
        data = data[:, [0, 1, 2, 3]]  # Keep only the first 4 columns and the 6th column of data
    assert data.shape[1] == 4, "Expected 3D points"
    print("Loaded %d points from %s" % (data.shape[0], filename))
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]
            #   PointField('v_r_comp', 16, PointField.FLOAT32, 1)]
    return pc2.create_cloud(header, fields, data)

def jpg_to_image(filename):
    img = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # 读取图像
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(img, encoding="bgr8")  # 将图像转换为 Image 消息
    return image_message

def main():
    rospy.init_node('txt_to_pointcloud2')
    pub_radar = rospy.Publisher('/radar_pc', PointCloud2, queue_size=10)
    pub_image = rospy.Publisher('/raw_img', Image, queue_size=10)
    rate = rospy.Rate(2) # 10hz
    root_path = '/home/ez/project/dataset/vod/view_of_delft_PUBLIC/'
    idx = 0
    while not rospy.is_shutdown():

        str_num = str(idx).zfill(5)
        idx += 1
        file_path_radar = root_path + "radar_5frames/training/velodyne/" + ('%s.bin' % str_num)
        file_path_radar = Path(file_path_radar)
        assert file_path_radar.exists(), "File not found: %s" % file_path_radar
        pointcloud2 = bin_to_pointcloud2(file_path_radar)
        pub_radar.publish(pointcloud2)
        print("Published pointcloud2 from %s" % file_path_radar)

        img_path = root_path + "lidar/training/image_2/" + ('%s.jpg' % str_num)  # 图像文件路径
        img_path = Path(img_path)
        assert img_path.exists(), "File not found: %s" % img_path

        image = jpg_to_image(img_path)  # 读取并转换图像

        pub_image.publish(image)  # 发布 Image 消息
        print("Published image from %s" % img_path)

        rate.sleep()

if __name__ == '__main__':
    main()