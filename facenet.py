from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.datasets import *
from utils.utils import *
from base_camera import BaseCamera
import time

min_person_size = 250
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


class Camera(BaseCamera):
    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        save_video=True
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter("result.mp4", fourcc, 15, (1280, 720))

        dataset = LoadStreams("0", img_size=640)
        for path, img, im0s, _ in dataset:
            fpstime = time.time()
            yolo_result, det = yolo_model.infer(path, img, im0s)
            # print("yolo_fps:", int(1 / (time.time() - fpstime)))
            if det is not None:
                aligned_test = []
                names2yolobox_index = []
                all_probs=[]
                all_boxes=[]
                #对每个有人的框检测人脸
                for index, (box_left, box_top, box_right, box_bottom, conf, label) in enumerate(det):
                    if box_bottom - box_top < min_person_size:
                        continue

                    _, probs, boxes, names, aligned = facenet.face_infer(
                        yolo_result[int(box_top):int(min(box_bottom, box_top + (box_right - box_left) * 0.5)),
                        int(box_left):int(box_right)])  # 把人的上面一部分放进去识别人脸
                    #对有人脸的行人框，把人脸信息加入列表
                    if boxes is not None:
                        aligned_test.extend(aligned)
                        names2yolobox_index.extend([index] * len(boxes))
                        all_probs.extend(probs)
                        all_boxes.extend(boxes)
                # 对每张图的所有人脸，检测是谁
                if len(aligned_test) > 0:
                    aligned_test = torch.stack(aligned_test).to(device)
                    embeddings_test = facenet.resnet(aligned_test).detach().cpu()
                    dists = [[(e1 - e2).norm().item() for e2 in facenet.embeddings] for e1 in embeddings_test]
                    names = [" " if min(dist) > 0.85 else facenet.dataset.idx_to_class[dist.index(min(dist))] for
                             dist in dists]
                    # print(" ".join(names))
                    # 画框 写名字
                    for index, name in enumerate(names):
                        box_left, box_top, box_right, box_bottom, conf, label = det[names2yolobox_index[index]]
                        if all_probs[index] > 0.8:
                            face_box=all_boxes[index]
                            cv2.putText(yolo_result[int(box_top):int(min(box_bottom, box_top + (box_right - box_left) * 0.5)), int(box_left):int(box_right)], name, (face_box[0], face_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
                            cv2.rectangle(yolo_result[int(box_top):int(min(box_bottom, box_top + (box_right - box_left) * 0.5)), int(box_left):int(box_right)], (face_box[0], face_box[1]), (face_box[2], face_box[3]),(0,255,0),2)
            # 写入mp4前缩小
            yolo_result = cv2.resize(yolo_result, (1280, 720))
            if save_video:
                vid_writer.write(yolo_result)
            print("\ryolo_face_fps:{}".format(int(1 / (time.time() - fpstime))))
            yield cv2.imencode('.jpg', yolo_result)[1].tobytes()


class Yolo:
    def __init__(self):
        weights = 'weights/yolov5m.pt'

        # Load model
        google_utils.attempt_download(weights)
        self.model = torch.load(weights, map_location=device)['model']

        self.model.to(device).eval()

        self.half = True
        if self.half:
            self.model.half()

        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def infer(self, path, img, im0s):
        img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5,
                                   fast=True, classes=[0], agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, self.names[int(c)])  # add to string

                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    if xyxy[3] - xyxy[1] < min_person_size:
                        continue
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                # names = [self.names[int(cls)] for clc in det[2]]

                return im0, det.cpu().numpy()
            else:
                return im0, None


class Facenet:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.workers = 0 if os.name == 'nt' else 4
        self.faces_load()

    def collate_fn(self, x):
        return x[0]

    def faces_load(self):
        self.dataset = datasets.ImageFolder('test_images_723')
        self.dataset.idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}
        self.loader = DataLoader(self.dataset, collate_fn=self.collate_fn, num_workers=self.workers)

        aligned = []
        names = []
        for x, y in self.loader:
            x_aligneds, probs, boxes = self.mtcnn(x, return_prob=True)
            if x_aligneds is not None:
                print('Face detected with probability: {:6f}'.format(probs[0]))
                aligned.append(x_aligneds[0])
                names.append(self.dataset.idx_to_class[y])
        print(" ".join(names))
        aligned = torch.stack(aligned).to(device)
        self.embeddings = self.resnet(aligned).detach().cpu()

    def face_infer(self, frame):
        aligned_test = []
        names = []
        # self.names_colors = []
        with torch.no_grad():
            faces, probs, boxes = self.mtcnn(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), return_prob=True)
            if boxes is not None:
                for index, face in enumerate(faces):
                    aligned_test.append(face)
                    # if probs[index] > 0.9:
                    #     font_color = (0, 255, 0)
                    # elif probs[index] > 0.7:
                    #     font_color = (0, 140, 255)
                    # elif probs[index] > 0.5:
                    #     font_color = (0, 0, 255)
                    # else:
                    #     font_color = (0, 0, 255)
                    # self.names_colors.append(font_color)
                # aligned_test = torch.stack(self.aligned_test).to(device)
                # embeddings_test = self.resnet(aligned_test).detach().cpu()
                #
                # dists = [[(e1 - e2).norm().item() for e2 in self.embeddings] for e1 in embeddings_test]
                # names = ["unknowed" if min(dist) > 1 else self.dataset.idx_to_class[dist.index(min(dist))] for dist in dists]
                # for index, name in enumerate(names):
                #     if probs[index]>0.8:
                #         cv2.rectangle(frame, (boxes[index][0], boxes[index][1]), (boxes[index][2], boxes[index][3]),
                #                       self.names_colors[index], 2)
                #         cv2.putText(frame, name, (boxes[index][0], boxes[index][1]), cv2.FONT_HERSHEY_COMPLEX, 1,
                #                     self.names_colors[index], 2)
        # cv2.imshow("face",frame)
        return frame, probs, boxes, names, aligned_test


yolo_model = Yolo()
facenet = Facenet()
