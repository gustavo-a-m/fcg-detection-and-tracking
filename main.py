'''
Descrição: Arquivo inicial do projeto
Autor: Gustavo Alves Moreira
Criação: 26/01/2020
'''

from argparse import ArgumentParser
import json
import math
import cv2
import numpy as np
import imutils.video

from timeit import time
from pose_estimation.pose_estimation import PoseEstimation
from pose_estimation.modules.input_reader import VideoReader, ImageReader
from tracking.tracking import Tracking
from f_formation.f_formation import FFormation 

def calculaPosicaoXY(pose_3d, idx):
    #posição 0 e 2 são Largura e Comprimento
    l_eye = pose_3d[15] 
    r_eye = pose_3d[17]
    if r_eye[0] == l_eye[0] or r_eye[1] == l_eye[1]:
        return None
    coef_angular_reta_normal = -((l_eye[0] - r_eye[0])/(l_eye[1] - r_eye[1]))
    arctan = np.arctan(coef_angular_reta_normal)
    if l_eye[1] < r_eye[1]:
        if arctan > 0:
            arctan = arctan + np.pi
        else:
            arctan = arctan - np.pi
    return [idx, (l_eye[0] + r_eye[0])/2, (l_eye[1] + r_eye[1])/2, arctan]

def calcDistUpBorder(x, y, xesq, xdir, ybox):
    xmedio = (xesq + xdir)/2
    dist = np.sqrt((x-xmedio)**2 + (y-ybox)**2)

    return dist


if __name__ == '__main__':
    parser = ArgumentParser(description='Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                            'The demo will look for a suitable plugin for device specified '
                            '(by default, it is GPU).',
                        type=str, default='CPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                            'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

    # if args.video == '' and args.images == '':
    #     raise ValueError('Either --video or --image has to be provided')
    
    # Cria novo objeto Pose Estimation
    _poseEstimation = PoseEstimation(args.use_openvino, args.device, args.extrinsics_path, args.images,
                                    args.video, args.fx, args.height_size)

    # Cria novo objeto Tracking
    _tracking = Tracking()

    # _images = []
    # for i in range(106, 111, 2):
    #     _images.append('000025/'+ str(i+25000).zfill(6) +'.jpg')
    # frame_provider = ImageReader(_images)

    frame_provider = ImageReader(args.images)

    if args.video != '':
        frame_provider = VideoReader(args.video)


    with open('output/resultado.txt', 'w') as arquivo:
        arquivo.write('TCC - Detecção e rastreamento de interações sociais\n')
        arquivo.write('Autor: Gustavo Alves Moreira\n')
        arquivo.write('Instituição: PUC Minas\n')
        arquivo.write('Data: Junho 2020\n\n\n')

    print("\n\n=============Finalizada Configuração inicial=============\n\n")
    esc_code = 27
    delay = 1
    
    frameCount = 1
    fps_imutils = imutils.video.FPS().start()
    fps_pose = 0.0
    fps_rastreamento = 0.0
    t_pose = 0.0
    t_rastreamento = 0.0

    pulaframe = 0

    for frame in frame_provider:
        print('\n' + str(frameCount))
        if frame is None:
            break
            
        # ======================================================================================
        t_pose = time.time()
        # Estima Poses
        poses_3d, edges, poses_2d = _poseEstimation.generatePoses(frame)

        poses_2d_Top_View = np.zeros((0, 4))
        poses_2d_Front_View = np.zeros((0, 2))
        for idx, pose in enumerate(poses_3d):
            __res = calculaPosicaoXY(pose, idx)
            if not (__res is None):
                poses_2d_Top_View = np.concatenate((poses_2d_Top_View, [__res]), axis=0)
                poses_2d_Front_View = np.concatenate((poses_2d_Front_View, [poses_2d[idx][0:2]]), axis=0)
                # r_p = poses_2d[idx][51:53]
                # l_p = poses_2d[idx][54:56]
                # if poses_2d[idx][53] < 0 or poses_2d[idx][56] < 0:
                #     r_p = poses_2d[idx][45:47]
                #     l_p = poses_2d[idx][48:50]
                #     print(r_p.astype(np.int))
                #     print(l_p.astype(np.int))
                # poses_2d_Front_View = np.concatenate((poses_2d_Front_View, [(r_p + l_p)/2]), axis=0)


        # Realizar o FFormation
        _FFormation = FFormation(poses_2d_Top_View, stride=35)
        # _FFormation.vis('teste')
        fps_pose = (fps_pose + (1./(time.time()-t_pose))) / 2


        # ======================================================================================
        t_rastreamento = time.time() 
        # Realiza o Tracking
        _tracking.track(frame)

        # Retorna as bounding boxes e o id do usuário identificado
        tracking_boxes = np.empty(shape=[0, 5]);
        for track in _tracking.tracker.tracks:
            tb = [];
            if track.is_confirmed() or track.time_since_update > 1:
                tb.append(track.track_id)
            else:
                tb.append(-1)
            
            tb = np.append(tb, track.to_tlbr(), axis=0)
            tracking_boxes = np.append(tracking_boxes, [tb], axis=0)

        # ======================================================================================
        result = np.empty(shape=[0, 4]);
        # Vincula a pose 2d ao id
        for idx, ps in enumerate(poses_2d_Front_View):
            min = float("inf")
            idxtrack = -1;
            for j, track in enumerate(tracking_boxes):
                if (((track[1] < track[3] and ps[0] > track[1] and ps[0] < track[3]) 
                    or (track[1] > track[3] and ps[0] < track[1] and ps[0] > track[3])) 
                    and ((track[2] < track[4] and ps[1] > track[2] and ps[1] < track[4]) 
                    or (track[2] > track[4] and ps[1] < track[2] and ps[1] > track[4]))):
                        dist = float("inf")
                        if track[2] > track[4]:
                            dist = calcDistUpBorder(ps[0], ps[1], track[1], track[3], track[2])
                        else:
                            dist = calcDistUpBorder(ps[0], ps[1], track[1], track[3], track[4])
                        
                        if dist < min:
                            min = dist
                            idxtrack = j

            if idxtrack > -1:
                result = np.append(result, [[tracking_boxes[idxtrack][0], _FFormation.est[0][idx], ps[0], ps[1]]], axis=0)
                tracking_boxes = np.delete(tracking_boxes, idxtrack, 0)
        
        fps_rastreamento = (fps_rastreamento + (1./(time.time()-t_rastreamento))) / 2

        # ======================================================================================
        if len(result) > 0:
            with open('output/resultado.txt', 'a') as arquivo:
                for i in result:
                    arquivo.write('Frame ' + str(frameCount) + '\t\t')
                    arquivo.write('Id ' + str(int(i[0])) + '\t')
                    arquivo.write('Grp ' + str(int(i[1])) + '\t')
                    arquivo.write('x ' + str(int(i[2])) + '\t')
                    arquivo.write('y ' + str(int(i[3])) + '\t\n')
                arquivo.write('\n\n');

        # _poseEstimation.plotPoses3D(poses_3d, edges)

        frameCount = frameCount + 1
        fps_imutils.update()
    
    fps_imutils.stop()
    with open('output/resultado.txt', 'a') as arquivo:
        arquivo.write('\nTempo total = ' + str(fps_imutils.elapsed()))
        arquivo.write('\nFPS Total = ' + str(fps_imutils.fps()));
        arquivo.write('\nFPS Pose Estimation = ' + str(fps_pose));
        arquivo.write('\nFPS Rastreamento = ' + str(fps_rastreamento))
    
    cv2.destroyAllWindows()
