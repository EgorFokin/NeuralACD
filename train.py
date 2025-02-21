# Some of the code is taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master CREDIT: Benny

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import coacd_modified
import pyntcloud
import json
import threading
import warnings


from pathlib import Path
from tqdm import tqdm
from utils.BaseUtils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

import model

warnings.filterwarnings('ignore')

coacd_modified.set_log_level("off")

os.environ["PYTHONHASHSEED"] = "0"

LR = 0.01
DECAY = 1e-4
DECAY_STEP = 10
LR_DECAY = 0.7
EPOCHS = 200

NUM_PLANES = 5

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def recalculate_planes(cmesh):
    planes = coacd_modified.best_cutting_planes(cmesh,num_planes=NUM_PLANES)
    planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
    return planes


def process_mesh(mesh, plane_cache):
    cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
    result = coacd_modified.normalize(cmesh)

    normalized_mesh = trimesh.Trimesh(result.vertices, result.indices)

    rotation = apply_random_rotation(normalized_mesh)

    mesh_hash = str(hash((mesh.vertices, mesh.faces)))
    if mesh_hash in plane_cache:
        planes = plane_cache[mesh_hash]
    else:
        print(f"Mesh {mesh_hash} not found in cache")
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes

    if len(planes) != NUM_PLANES:
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes
    
    try:
        target = []
        for plane in planes:
            a, b, c, d = apply_rotation_to_plane(*plane[:4], rotation)
            target.append([a, b, c, d])
                
    except Exception as e:
        print(e)
        #try again
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes

        try:
            target = []
            for plane in planes:
                a, b, c, d = apply_rotation_to_plane(*plane[:4], rotation)
                target.append([a, b, c, d])
        except:
            mesh.export("broken_mesh.obj")
            return None, None
        
    if len(target) != NUM_PLANES:
        mesh.export("broken_mesh.obj")
        return None, None

    normalized_mesh.export(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"), vertex_normal=True)
    pc_mesh = pyntcloud.PyntCloud.from_file(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"))

    os.remove(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"))
    pc = pc_mesh.get_sample("mesh_random", n=512, normals=False, as_PyntCloud=True)

    return pc.points, target

def preprocess_data(batch):
    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)

    points = []
    target = []

    for mesh in batch:
        points_, planes = process_mesh(mesh, plane_cache)
        if points_ is not None:
            points.append(points_)
            target.append(planes)
        else:
            print("Skipping batch due to broken data")
            return None, None 

    points = np.array(points)
    points = torch.Tensor(points)
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    points = points.float().cuda()

    target = np.array(target)
    target = torch.Tensor(target)
    target = target.float().cuda()

    def write_to_cache(plane_cache):
        with open("plane_cache.json", "w") as plane_cache_f:
            json.dump(plane_cache, plane_cache_f)

    t = threading.Thread(target=write_to_cache, args=(plane_cache,)) #prevents KeyboardInterrupt during write
    t.start()
    t.join()

    return points, target



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth'))

def main():
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')


    '''MODEL LOADING'''

    num_output = 4

    shutil.copy('model/model.py', str(exp_dir))
    shutil.copy('model/pointnet2_utils.py', str(exp_dir))

    predictor = model.get_model(num_output).cuda()
    criterion = model.get_loss().cuda()
    #predictor.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # try:
    #     checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = 0
    #     #start_epoch = checkpoint['epoch']
    #     checkpoint_state_dict = checkpoint['model_state_dict']

    #     # remove layers not used in the model
    #     del checkpoint_state_dict['fc1.weight']
    #     del checkpoint_state_dict['fc1.bias']
    #     del checkpoint_state_dict['bn1.weight']
    #     del checkpoint_state_dict['bn1.bias']
    #     del checkpoint_state_dict['bn1.running_mean']
    #     del checkpoint_state_dict['bn1.running_var']
    #     del checkpoint_state_dict['bn1.num_batches_tracked']
    #     del checkpoint_state_dict['fc2.weight']
    #     del checkpoint_state_dict['fc2.bias']
    #     del checkpoint_state_dict['bn2.weight']
    #     del checkpoint_state_dict['bn2.bias']
    #     del checkpoint_state_dict['bn2.running_mean']
    #     del checkpoint_state_dict['bn2.running_var']
    #     del checkpoint_state_dict['bn2.num_batches_tracked']
    #     del checkpoint_state_dict['fc3.weight']
    #     del checkpoint_state_dict['fc3.bias']

    #     predictor.load_state_dict(checkpoint_state_dict, strict=False)
    #     log_string('Use pretrain model')
    # except Exception as e:
    #     print(e)
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0
    #     predictor = predictor.apply(weights_init)

    start_epoch = 0
    #predictor = predictor.apply(weights_init)


    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-08
    )


    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = DECAY_STEP

    print('Using', torch.cuda.get_device_name(torch.cuda.current_device()))

    best_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):

        log_string('Epoch %d (%d/%s):' %
                   (epoch + 1, epoch + 1, EPOCHS))
        '''Adjust learning rate and BN momentum'''
        lr = max(LR * (LR_DECAY **
                 (epoch // DECAY_STEP)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        predictor = predictor.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        predictor = predictor.train()

        '''learning one epoch'''
        i = 0
        batch_size = 16
        batch = []

        running_tloss = 0

        for mesh in load_shapenet(debug=True, data_folder="data/ShapenetRedistributed"):
            batch.append(mesh)
            if len(batch) != batch_size:
                continue

            
            optimizer.zero_grad()
            
            start_time = datetime.datetime.now()
            points, target = preprocess_data(batch)
            print("Preprocessing time:", datetime.datetime.now()-start_time)

            if points is None:
                batch = []
                continue

            points = points.transpose(2, 1)

            start_time = datetime.datetime.now()
            seg_pred = predictor(
                points)
            
            print(target)
            print(seg_pred)
            
            loss = criterion(seg_pred, target)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient of {name}: {param.grad.norm()}")
            optimizer.step()
            print("Training time:", datetime.datetime.now()-start_time)
            print('Training loss:', running_tloss/(i+1))
            running_tloss += loss.item()
            i+=1
            print(f"{i*batch_size} meshes processed")

            batch = []

        

        #validation
        print("Running validation...")
        predictor = predictor.eval()
        running_vloss = 0
        i = 0
        with torch.no_grad():
            batch = []
            for mesh in load_shapenet(debug=False, data_folder="data/ShapenetRedistributed_val"):
                batch.append(mesh)
                if len(batch) != batch_size:
                    continue

                points, target = preprocess_data(batch)

                if points is None:
                    batch = []
                    continue

                points = points.transpose(2, 1)

                seg_pred, trans_feat = predictor(
                    points)
                
                loss = criterion(seg_pred,target,trans_feat)
                running_vloss += loss.item()
                i+=1
                print(f"{i*batch_size} val meshes processed")
                batch = []
        

        print('Validation loss:', running_vloss/i)
        best = False
        if running_vloss < best_loss:
            best_loss = running_vloss
            best = True


        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best, str(exp_dir) + '/checkpoints/', 'checkpoint.pth')


if __name__ == '__main__':
    main()
