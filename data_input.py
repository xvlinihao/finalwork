import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset

def read_xyz(filename):
    """读取点云数据"""
    data=PlyData.read(filename)['vertex']
    xyz=np.c_[data['x'],data['y'],data['z']]
    return xyz

def generate_data(xyz):
    data=[]
    theta_x=np.random.uniform(-180,180)
    theta_y=np.random.uniform(-180,180)
    theta_z=np.random.uniform(-180,180)
    move_x=np.random.uniform(-0.1,0.1)
    move_y=np.random.uniform(-0.1,0.1)
    move_z=np.random.uniform(-0.1,0.1)
    rotate_x=np.array([
        [1,0,0],
        [0,np.cos(theta_x),np.sin(theta_x)],
        [0,-np.sin(theta_x),np.cos(theta_x)]
    ])
    rotate_y=np.array([
        [np.cos(theta_y),0,-np.sin(theta_y)],
        [0,1,0],
        [np.sin(theta_y),0,np.cos(theta_y)]
    ])
    rotate_z=np.array([
        [np.cos(theta_z),np.sin(theta_z),0],
        [-np.sin(theta_z),np.cos(theta_z),0],
        [0,0,1]
    ])
    drop_rate=np.random.uniform(0,0.20)
    for item in xyz:
        item+=[move_x,move_y,move_z]
        item=np.array(item)
        item=np.dot(item,rotate_x)
        item=np.dot(item,rotate_y)
        item=np.dot(item,rotate_z)
        item=item.tolist()
        drop_flag=np.random.uniform(0,1)
        if(drop_flag>drop_rate):
            data.append(item)
        else:
            data.append([0,0,0])
        #data.append(item)
    data=np.array(data)
    return data




class RegisterSet(Dataset):
    def __init__(self,src,target):
        #self.src=read_xyz(src)
        self.src=np.load(src)
        self.target=np.load(target)
        #self.data=resize(self.src)
        #factor=self.src.shape[0]//2048
        #points_num=factor*2048
        #self.src=self.src[:points_num]
       # self.target=self.target[:points_num]
        """for i in range(data_num):
            new_data=generate_data(self.src)
            np.save('transformed_data/t{}.npy'.format{i})
            self.data.append(new_data)"""
        #self.new_data=generate_data(self.src)
        #self.pointcloud1=np.array_split(self.src,factor,axis=0)
        #self.pointcloud2=np.array_split(self.target,factor,axis=0)
        #self.pointcloud1=resize(np.random.permutation(pointcloud1))
        #self.pointcloud2=resize(np.random.permutation(new_data))
    def __getitem__(self,item):
        #pcl1=np.random.permutation(self.pointcloud1[item].T)
        #pcl2=np.random.permutation(self.pointcloud2[item].T)
        #pcl1=self.pointcloud1[item].T
        #pcl1=np.random.permutation(pcl1)
        #pcl2=self.pointcloud2[item].T
        #pcl2=np.random.permutation(pcl2)           
        return self.src.T.astype('float32'),self.target.T.astype('float32')
        #return self.src.T.astype('float32'),self.new_data.T.astype('float32')
    def __len__(self):
        #return self.src.shape[0]//2048
        return 1