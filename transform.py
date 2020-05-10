import numpy as np
from plyfile import PlyData,PlyElement
import random
import plotly.graph_objects as go
import os
import argparse


def read_xyz(filename):
    """读取点云坐标数据"""
    data=PlyData.read(filename)['vertex']
    xyz=np.c_[data['x'],data['y'],data['z']]
    return xyz

"""def generate_dataset(xyz):
    data=[]
    theta_x=random.uniform(-180,180)
    theta_y=random.uniform(-180,180)
    theta_z=random.uniform(-180,180)
    move_x=random.uniform(-0.1,0.1)
    move_y=random.uniform(-0.1,0.1)
    move_z=random.uniform(-0.1,0.1)
    rotate_x=np.array([[1,0,0],[0,np.cos(theta_x),np.sin(theta_x)],[0,-np.sin(theta_x),np.cos(theta_x)]])
    rotate_y=np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])
    rotate_z=np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    for item in xyz:
        item+=[move_x,move_y,move_z]
        item=np.array(item)
        item=np.dot(item,rotate_x)
        item=np.dot(item,rotate_y)
        item=np.dot(item,rotate_z)
        item=item.tolist()
        drop_flag=random.uniform(0,0.20)
        drop_rate=random.uniform(0,1)
        if(drop_flag>drop_rate):
            data.append(item)
    data=np.array(data)
    return data"""

def generate_dataset(xyz):
    """随机平移和旋转并dropout一些点，制造数据集"""
    data=[]
    theta_x=random.uniform(-180,180)
    theta_y=random.uniform(-180,180)
    theta_z=random.uniform(-180,180)
    move_x=random.uniform(-0.1,0.1)
    move_y=random.uniform(-0.1,0.1)
    move_z=random.uniform(-0.1,0.1)
    rotate_x=np.array([[1,0,0],[0,np.cos(theta_x),np.sin(theta_x)],[0,-np.sin(theta_x),np.cos(theta_x)]])
    rotate_y=np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])
    rotate_z=np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    for item in xyz:
        item+=[move_x,move_y,move_z]
        item=np.array(item)
        item=np.dot(item,rotate_x)
        item=np.dot(item,rotate_y)
        item=np.dot(item,rotate_z)
        item=item.tolist()
        """drop_flag=random.uniform(0,0.20)
        drop_rate=random.uniform(0,1)
        if(drop_flag>drop_rate):
            data.append(item)"""
        data.append(item)
    data=np.array(data)
    result=np.array(drop_out(data))
    return result

def drop_out(xyz):
    "用于随机丢弃一些连续点"
    start=np.random.randint(0,np.size(xyz,0))
    end=np.random.randint(0,np.size(xyz,0))
    
    data=[]
    if start>end:
        start,end=end,start
    idx=0
    for item in xyz:
        idx+=1
        if idx<=start or idx>=end:
            data.append(item)
        else:
            data.append(np.array([0,0,0]))
    return data
    

def main():
    parser = argparse.ArgumentParser(description='Input data')
    parser.add_argument('--data_sort',type=str,default='npy',metavar='N',
    help='the sort of input data, npy or ply')
    parser.add_argument('--src',type=str,default='',metavar='N',
    help='the path of input data')

    args = parser.parse_args()
    if args.data_sort=='npy':
        origin_data=np.load(args.src)
    else:
        origin_data = read_xyz(args.src)
    new_data=[]
    for _ in range(8):
        new_data.append(generate_dataset(origin_data))
    idx=1
    if not os.path.exists('transformed_data'):
        os.makedirs('transformed_data')
    for _ in new_data:
        np.save('transformed_data/{}.npy'.format(idx),_)
        idx=idx+1

if __name__ == "__main__":
    main()


