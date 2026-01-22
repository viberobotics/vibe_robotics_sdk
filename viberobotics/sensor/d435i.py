import pyrealsense2 as rs

import numpy as np
import cv2
import open3d as o3d

class RealsenseD435i:
    def __init__(self,
                 clipping_distance=1. # meters
                 ):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        for dev in device.sensors:
            print(f'Found sensor: {dev.get_info(rs.camera_info.name)}')

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(self.config)
        
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth Scale is: {self.depth_scale}")
        self.clipping_distance = clipping_distance / self.depth_scale
        
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self.colorizer = rs.colorizer()
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        
    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        return depth_frame, color_frame
    
    def get_aligned_images(self):
        depth_frame, color_frame = self.get_aligned_frames()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
        
    def remove_bg(self, depth_image, color_image, grey_color=153):
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        return bg_removed
    
    def get_pointcloud(self):
        depth_frame, color_frame = self.get_aligned_frames()
        depth_frame = self.decimate_filter.process(depth_frame)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = color_frame.get_width(), color_frame.get_height()
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        color_image = np.asanyarray(color_frame.get_data())
        # get colors vectorize
        u = (texcoords[:, 0] * w).astype(np.int32)
        v = (texcoords[:, 1] * h).astype(np.int32)
        colors = color_image[v, u] / 255
        return verts, colors[:, [2, 1, 0]]  # BGR to RGB
        
if __name__ == '__main__':
    mode = input('select mode: (1) aligned images (2) pointcloud:')
    mode = int(mode)
    
    if mode == 1:
        cam = RealsenseD435i()
        while True:
            depth_image, color_image = cam.get_aligned_frames()
            if depth_image is None or color_image is None:
                continue
            grey_color = 153
            bg_removed = cam.remove_bg(depth_image, color_image, grey_color)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    elif mode == 2:
        cam = RealsenseD435i()
        vis = o3d.visualization.Visualizer()
        vis.create_window("Realsense D435i Pointcloud", width=640, height=480)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
        vis.add_geometry(pcd)
        # display xyz frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        vis.add_geometry(coord_frame)
        while vis.poll_events():
            verts, colors = cam.get_pointcloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()