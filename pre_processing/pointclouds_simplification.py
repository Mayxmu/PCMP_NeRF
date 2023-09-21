"""
This file help you simplify point clouds via sub-sampling.

scene: scene name, e.g. 'ScanNet/scene0010_00'
input_path: input .ply path, need to be simplified
output_path: output simplified .ply path.
side_len: voxel side length, use 0.01m in this paper.
num_in_each_voxel: predefined number of points in each voxel, can be changed.

Note: if the scene is too large for the hardware, try to divide whole point cloud into small parts and simplify them seperately.

"""
from __future__ import division
import cv2, time, random, math, os
import numpy as np

import linecache


class uniform_sampler:

    def __init__(self, side_len, num_in_each_box, ply_path, output_path):

        self.side_len = side_len
        self.ply_path = ply_path
        self.num_in_each_box = num_in_each_box
        self.position_color = []
        self.header = []
        self.num_unit_box = 0
        self.num_point_before = 0
        self.output_path = output_path
        self.num_points = 0
        # self.wstart = 0


    def loadfile(self,rstart,rlines):

        st = time.time()
        ply = []

        # file = open(self.ply_path)
        t = rstart


        while t < rstart + rlines:

            line = linecache.getline(self.ply_path, t).strip('\n')
            # line = file.readline().strip('\n')
            if not line:
                break
            line = line.split(' ')
            ply.append(np.array(line))
            # print(t)
            # print(line)
            t = t + 1
        # file.close()

        cnt = 0
        for i in range(len(ply)):
            line = ply[i]
            if line[0] == 'end_header':
                break
            cnt = cnt + 1
            print(cnt)

        self.header = ply[0:cnt+1]
        self.position_color = ply[cnt+1:]
        self.num_point_before = len(self.position_color)
        self.position_color = np.array(self.position_color)

        print('load time: %s' % (time.time() - st))

        return self.header, self.position_color


    def write(self, point_clouds_simplified):

        file = open(self.output_path, 'a')
        self.num_points = self.num_points + point_clouds_simplified.shape[0]

        num_elements = point_clouds_simplified.shape[1]

        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %s\n' % (self.num_points))
        file.write('property float32 x\n')
        file.write('property float32 y\n')
        file.write('property float32 z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('end_header\n')


        print(point_clouds_simplified.shape)
        print(type(point_clouds_simplified))
        print(type(point_clouds_simplified[1]))
        print(type(point_clouds_simplified[21,2]))
        print(type(point_clouds_simplified[21,4]))

        for i in range(point_clouds_simplified.shape[0]):
            for j in range(num_elements):

                file.write(point_clouds_simplified[i, j])
                file.write(' ')

            file.write('\n')

        file.close()


    # Voxelize 3D space and calculate number of points in each voxel.
    def bounding_box(self):

        st = time.time()
        x = np.float32(self.position_color[:,0])
        y = np.float32(self.position_color[:,1])
        z = np.float32(self.position_color[:,2])

        max_x = np.max(x)
        min_x = np.min(x)
        diff_x = max_x - min_x
        max_y = np.max(y)
        min_y = np.min(y)
        diff_y = max_y - min_y
        max_z = np.max(z)
        min_z = np.min(z)
        diff_z = max_z - min_z

        num_x = np.max(np.rint((x - min_x) / self.side_len)) + 1
        num_y = np.max(np.rint((y - min_y) / self.side_len)) + 1
        num_z = np.max(np.rint((z - min_z) / self.side_len)) + 1

        # total num of voxels. Maybe too many voxels for very large scene.
        # If out of source, try to divide whole point clouds into small parts, and simplify them separately.
        self.num_unit_box = num_x * num_y * num_z

        print('num_box_x: %s, num_box_y: %s, num_box_z: %s, num_box_total: %s' % (num_x, num_y, num_z, self.num_unit_box))

        top_left = [min_x, max_y, min_z]
        top_right = [max_x, max_y, min_z]
        bottom_left = [min_x, min_y, min_z]
        bottom_right = [min_x, max_y, min_z]

        index = np.rint((x - min_x)/self.side_len) + np.rint((y - min_y)/self.side_len)*num_x + np.rint((z - min_z)/self.side_len)*num_x*num_y
        index1 = np.argsort(index)
        index = index[index1]
        self.position_color = self.position_color[index1, :]
        select_probabilities = np.zeros(len(index))   # the probability of each point can be selected.

        cnt_l = 0
        num = 0
        for i in range(int(self.num_point_before)):

            cnt_r = i + 1
            tmp = index[i]
            if cnt_r < self.num_point_before:
                if not (index[cnt_r] == tmp):
                    select_probabilities[cnt_l: cnt_r] = self.num_in_each_box/(cnt_r - cnt_l)
                    num = num + (cnt_r - cnt_l)
                    cnt_l = cnt_r
            else:
                select_probabilities[cnt_l: cnt_r] = self.num_in_each_box/(cnt_r - cnt_l)
                num = num + (cnt_r - cnt_l)


        assert num == self.num_point_before

        print('voxelize_time: %s' % (time.time() - st))

        return select_probabilities, bottom_left


    # randomly select points.
    def sampling(self):

        st = time.time()

        select_probabilities, start_point = self.bounding_box()
        position_color_after = np.zeros_like(self.position_color)
        random = np.random.uniform(0, 1, self.num_point_before)  # generate sampling probability.

        cnt = 0

        for i in range(self.num_point_before):

            select_probability = select_probabilities[i]  # the probability of the point can be selected

            if select_probability > random[i]:
                position_color_after[cnt, :] = self.position_color[i, :]
                cnt = cnt + 1
        position_color_after = position_color_after[0:cnt, :]

        print(position_color_after.shape)

        self.write(point_clouds_simplified=position_color_after)


        print('simplification_time: %s' % (time.time() - st))

        return position_color_after







if __name__ == '__main__':

    scene = 'ScanNet/scene0017_00'
    # scene = 'Matterport3D/29hnd4uzFmX'
    # input_path = '../pre_processing_results/%s/point_clouds_simplified_11.ply' % scene
    # output_path = '../pre_processing_results/%s/point_clouds_simplified_12.ply' % scene
    input_path = '../pre_processing_results/%s/point_clouds_simplified_2.ply' % scene
    output_path = '../pre_processing_results/%s/point_clouds_simplified_3.ply' % scene
    side_len = 0.05  # voxel side length 0.01
    num_in_each_voxel = 20 # Predefined number of points in each voxel.30

    if os.path.isfile(input_path):
        sampler = uniform_sampler(side_len, num_in_each_voxel, input_path, output_path)
        ##[1,30,000,000],[30,000,001,60,000,000],[60,000,001,90,000,000],[90,000,001,104,256,042]
        # # 第五次简化
        # # [1,14378197]
        # side_len = 0.02  # voxel side length
        # num_in_each_voxel = 20  # Predefined number of points in each voxel.

        # # 第六次简化
        # # [1,7607860]
        # side_len = 0.02  # voxel side length
        # num_in_each_voxel = 20  # Predefined number of points in each voxel.

        # # 第七次简化
        # # [1,7607860]
        # side_len = 0.03  # voxel side length
        # num_in_each_voxel = 15  # Predefined number of points in each voxel.


        # # 第八次简化
        # # [1,4513794]
        # side_len = 0.05  # voxel side length
        # num_in_each_voxel = 15  # Predefined number of points in each voxel.

        # # 第九次简化
        # # [1,2191483]
        # side_len = 0.05  # voxel side length
        # num_in_each_voxel = 5 # Predefined number of points in each voxel.

        # # 第十次简化
        # # [1,869759]
        # side_len = 0.05  # voxel side length
        # num_in_each_voxel = 3 # Predefined number of points in each voxel.



        # 剩 537659
        # # 第十一次简化
        # # [1,537659]
        # side_len = 0.1  # voxel side length
        # num_in_each_voxel = 2 # Predefined number of points in each voxel.

        # # 第十二次简化
        # # [1,351921]
        # side_len = 0.1  # voxel side length
        # num_in_each_voxel = 1 # Predefined number of points in each voxel.


        sampler.loadfile(rstart=1, rlines=7238634)
        sampler.sampling()
    else:
        print('Missing .ply file!')




