import cv2
import imageio
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageDraw
import numpy as np


def get_transform():
    transform = Compose([
        Resize((896, 896)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

# 定义加载模型的函数
def load_model_for_inference(path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def inference(model, image, score_threshold_bi, score_threshold_car):
    pred = {}
    transform = get_transform()
    img_transformed = transform(image).unsqueeze(0)
    img_transformed = img_transformed.to(device)
    with torch.no_grad():
        predictions = model(img_transformed)
    predictions = predictions[0]
    
    # 获取所有预测的标签、得分和框
    all_labels = predictions['labels'].cpu().numpy()
    all_scores = predictions['scores'].cpu().numpy()
    all_boxes = predictions['boxes'].cpu().numpy()
    
    # 初始化空列表来保存过滤后的结果
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    # 对每个预测进行检查并应用相应的阈值
    for label, score, box in zip(all_labels, all_scores, all_boxes):
        if label == 1 and score > score_threshold_bi:
            filtered_labels.append(label)
            filtered_scores.append(score)
            filtered_boxes.append(box)
        elif label != 1 and score > score_threshold_car:
            filtered_labels.append(label)
            filtered_scores.append(score)
            filtered_boxes.append(box)
    
    # 将过滤后的结果保存到pred字典中
    pred['boxes'] = np.array(filtered_boxes)
    pred['scores'] = np.array(filtered_scores)
    pred['labels'] = np.array(filtered_labels)
    print(pred['scores'])
    return pred


def visualize_frame(frame, boxes, scores, labels, trackers):
    # 绘制检测框
    # for box, score, label in zip(boxes, scores, labels):
    #     cls, color = '', (255, 255, 255)  # 默认颜色为白色
    #     if label == 1:
    #         cls, color = 'bicycle', (150, 123, 238)  # 自行车，紫色
    #     elif label == 2:
    #         cls, color = 'car', (123, 238, 176)  # 车辆，淡绿色
        
    #     x1, y1, x2, y2 = [int(coord) for coord in box]
    #     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(frame, f'{cls}:{score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    for trk in trackers:
        # 取出跟踪器中存储的最后一个边界框
        if 'last_box' in trk:
            box = trk['last_box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # 根据跟踪器类型选择颜色
            color = (255, 255, 255)  # 默认白色
            cls = ''
            if trk['type'] == 1:
                cls, color = 'bicycle', (150, 123, 238)
            elif trk['type'] == 2:
                # cls, color = 'bicycle', (150, 123, 238)   ## people/ bycycle
                cls, color = 'car', (123, 238, 176)
            
            # 绘制边界框
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, cls, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # # 绘制跟踪器的预测位置和历史轨迹
    # for trk in trackers:
    #     # 根据跟踪器类型选择颜色
    #     color = (255, 255, 255)  # 默认白色
    #     cls = ''  # 初始化类别名称
    #     if trk['type'] == 1:
    #         cls, color = 'bicycle', (150, 123, 238)  # 自行车，紫色
    #     elif trk['type'] == 2:
    #         cls, color = 'car', (123, 238, 176)  # 车辆，淡绿色
        
        # 绘制历史轨迹点
        if 'history' in trk:
            for pt in trk['history'][-60:]:  # 只取最近的100个历史点
                cv2.circle(frame, pt, 3, color, -1)  # 使用类型特定的颜色绘制轨迹点
                
        # 绘制当前预测位置
        pred_x, pred_y = int(trk['kf'].x[0]), int(trk['kf'].x[1])
        cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)  # 使用相同的颜色绘制当前预测位置

    return frame

def adjust_boxes(boxes, transformed_dim=(896, 896), new_dim=(1280, 720)):
    # 计算宽度和高度的缩放比例
    scale_w, scale_h = new_dim[0] / transformed_dim[0], new_dim[1] / transformed_dim[1]
    
    adjusted_boxes = []
    for box in boxes:
        # 缩放边界框坐标
        x1, y1, x2, y2 = box
        adjusted_box = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]
        adjusted_boxes.append(adjusted_box)
    
    return adjusted_boxes

def adjust_frame_size(frame, target_size=(1280, 720), macro_block_size=16):
    # 确保目标尺寸是16的倍数
    target_width = int(np.ceil(target_size[0] / macro_block_size) * macro_block_size)
    target_height = int(np.ceil(target_size[1] / macro_block_size) * macro_block_size)
    
    # 确保 frame 是一个 NumPy 数组
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    
    # 调整帧尺寸
    adjusted_frame = cv2.resize(frame, (target_width, target_height))
    
    return adjusted_frame


def initialize_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 状态: [x, y, dx, dy], 观测: [x, y]
    dt = 1.0  # time gap

    # 状态转移矩阵
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 测量矩阵
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.P *= 1000.  # 初始状态协方差
    kf.R = np.eye(2) * 10  # 测量噪声
    kf.Q = np.eye(4) * 0.1  # 过程噪声

    return {'kf': kf, 'missed_count': 0, 'history': []}

def compute_iou(boxA, boxB):
    # 计算交集的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算各自的边界框面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并返回IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compute_cost_matrix(detections, trackers,adjusted_boxes_resized):
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if 'last_box' in trk:
                # 如果跟踪器有最后一个边界框，计算与检测的IoU作为成本
                iou = compute_iou(adjusted_boxes_resized[d], trk['last_box'])
                cost_matrix[d, t] = 1 - iou  # 1-IoU作为成本，因为我们想要最大化IoU
            else:
                # 使用位置信息计算成本，如果没有边界框信息
                cost_matrix[d, t] = np.linalg.norm(np.array(det[:2]) - np.array(trk['kf'].x[:2].reshape(-1)))
    return cost_matrix

def assign_detections_to_trackers(detections, trackers,adjusted_boxes_resized):
    cost_matrix = compute_cost_matrix(detections, trackers,adjusted_boxes_resized)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix

def update_trackers(trackers, adjusted_boxes_resized, pred_scores, row_ind, col_ind, pred_labels):
    matched_indices = set(row_ind)
    matched_trackers = set(col_ind)
    detections = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in adjusted_boxes_resized]

    # 首先更新所有匹配到的跟踪器
    for d, t in zip(row_ind, col_ind):
        trk = trackers[t]
        det = detections[d]
        trk['kf'].update(np.array([[det[0]], [det[1]]]))  # 更新卡尔曼滤波器
        trk['missed_count'] = 0  # 重置missed_count
        trk['last_box'] = adjusted_boxes_resized[d]
        trk['type'] = pred_labels[d]
        # 注意：我们不在这里更新历史轨迹，而是在预测步骤中统一更新

    # 处理未匹配的检测，评估是否创建新的跟踪器
    for d in range(len(detections)):
        if d not in matched_indices:  # 检测到的目标没有匹配的跟踪器
            det = detections[d]
            should_create_new_tracker = True  # 假定我们决定创建新的跟踪器
            if should_create_new_tracker:
                kf = initialize_kalman()
                kf['kf'].update(np.array([[det[0]], [det[1]]]))  # 使用检测初始化卡尔曼滤波器
                new_tracker = {
                    'kf': kf['kf'],
                    'missed_count': 0,
                    'last_box':adjusted_boxes_resized[d],
                    # trk['last_box'] = det
                    'history': [(int(det[0]), int(det[1]))],
                    'type': pred_labels[d]  # 保存目标类型
                }
                trackers.append(new_tracker)

    # 增加未匹配跟踪器的missed_count
    for t, trk in enumerate(trackers):
        if t not in matched_trackers:
            trk['missed_count'] += 1



import cv2
import imageio
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_for_inference('checkpoints_1/epoch_9.pth',device)
# pred=inference(model,image,0.25)
# 假设其他必要的函数（如inference, initialize_kalman, adjust_boxes等）已经定义

video_path = 'videos/Drone.mp4'
output_filename = 'videos/output_video_skip.mp4'
video_reader = imageio.get_reader(video_path)
writer = imageio.get_writer(output_filename, fps=20)

trackers = []
frame_index=0
for frame in video_reader:
    original_dim = (896, 896)  #  Size of object detection image after transform
    frame_resized = adjust_frame_size(frame, target_size=(1280, 720))   ## resize frame


    frame_pil = Image.fromarray(frame_resized)
    pred = inference(model, frame_pil, 0.35,0.9)  # 执行目标检测
    pred_boxes, pred_scores, pred_labels = pred['boxes'], pred['scores'], pred['labels']
    
    if len(pred_boxes) > 0:
        adjusted_boxes_resized = adjust_boxes(pred_boxes, original_dim, (1280, 720))
        detections = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in adjusted_boxes_resized]
        row_ind, col_ind, _ = assign_detections_to_trackers(detections, trackers,adjusted_boxes_resized)
        update_trackers(trackers, adjusted_boxes_resized, pred_scores, row_ind, col_ind, pred_labels)  # 使用检测结果更新跟踪器状态
    else:
        writer.append_data(frame_resized)
        continue

    
     # Predict 执行预测步骤并更新跟踪器的历史轨迹
    for trk in trackers:
        trk['kf'].predict()  # 执行预测
        pred_x, pred_y = int(trk['kf'].x[0]), int(trk['kf'].x[1])
        trk['history'].append((pred_x, pred_y))  # 更新历史轨迹

    trackers = [trk for trk in trackers if trk['missed_count'] < 50]  # 清理长时间未更新的跟踪器

    vis_frame = visualize_frame(frame_resized, adjusted_boxes_resized, pred_scores, pred_labels, trackers)
    output_temp=f"videos/temp/{frame_index}.jpg"
    cv2.imwrite(output_temp,vis_frame)
    writer.append_data(vis_frame)  # 写入可视化后的帧

    frame_index += 1
    # if frame_index > 50:
    #     break
    print(frame_index )

writer.close()