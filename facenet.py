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
                facenet_input=[]
                aligned_test= []
                person_img_boxes=[]
                person_img_bestprobs = []
                #对每个有人的框检测人脸
                for index, (box_left, box_top, box_right, box_bottom, conf, label) in enumerate(det):
                    if box_bottom - box_top < min_person_size:
                        continue
                    face_prebox=yolo_result[int(box_top):int(min(box_bottom, box_top + (box_right - box_left) * 0.7)),int(box_left):int(box_right)]
                    # print(face_prebox.shape)
                    face_prebox=cv2.resize(face_prebox,(250,int(250/face_prebox.shape[1]*face_prebox.shape[0])))
                    if face_prebox.shape[0]>200:
                        face_prebox=face_prebox[0:200,:]
                    else:
                        face_prebox=np.vstack((face_prebox,np.zeros((200-face_prebox.shape[0],250,3), np.uint8)))
                    # print(face_prebox.shape)

                    # cv2.imshow("face_back",face_prebox)
                    # cv2.waitKey(1)
                    facenet_input.append(Image.fromarray(cv2.cvtColor(face_prebox, cv2.COLOR_BGR2RGB)))
                    person_img_boxes.append([box_left, box_top, box_right, box_bottom])
                _, probs, boxes, faces = facenet.face_infer(facenet_input)  # 把人的上面一部分放进去识别人脸
                # print("=========boxes=========\n",boxes)
                # print("=========probs=========\n",probs)
                # 对有人脸的行人框，把人脸信息加入列表
                for index, pre_person_img_prob in enumerate(probs):
                    if boxes[index] is not None:#这个人框里面有人脸
                        pre_person_bestface_index=np.argmax(pre_person_img_prob)#找置信度最大的框
                        aligned_test.append(faces[index][pre_person_bestface_index])
                        person_img_bestprobs.append(max(pre_person_img_prob))
                # 对每张图的所有人脸，检测是谁
                if len(aligned_test) > 0:
                    aligned_test = torch.stack(aligned_test).to(device)
                    embeddings_test = facenet.resnet(aligned_test).detach().cpu()
                    dists = [[(e1 - e2).norm().item() for e2 in facenet.embeddings] for e1 in embeddings_test]
                    names = ["?" if min(dist) > 0.85 else facenet.dataset.idx_to_class[dist.index(min(dist))] for
                             dist in dists]
                    # print(" ".join(names))
                    # 画框 写名字
                    for index, name in enumerate(names):
                        person_img_box = person_img_boxes[index]
                        c1, c2 = (int(person_img_box[0]), int(person_img_box[1])), (
                        int(person_img_box[2]), int(person_img_box[3]))
                        tl = 10
                        color = (255,69,0)
                        cv2.rectangle(yolo_result, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        if person_img_bestprobs[index]>0.7:
                            if name!='?':
                                tf = max(tl - 1, 1)  # font thickness
                                t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
                                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                cv2.rectangle(yolo_result, c1, c2, color, -1, cv2.LINE_AA)  # filled
                                cv2.putText(yolo_result, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                            lineType=cv2.LINE_AA)
            # 写入mp4前缩小
            yolo_result = cv2.resize(yolo_result, (1280, 720))
            if save_video:
                vid_writer.write(yolo_result)
            print("\ryolo_face_fps:{}".format(int(1 / (time.time() - fpstime))))
            yield cv2.imencode('.jpg', yolo_result)[1].tobytes()


class Yolo:
    def __init__(self):
        weights = 'weights/yolov5s.pt'

        # Load model
        google_utils.attempt_download(weights)
        self.model = torch.load(weights, map_location=device)['model']

        self.model.to(device).eval()

        self.half = True
        if self.half:
            self.model.half()

        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names

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
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
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
        # self.names_colors = []
        with torch.no_grad():
            faces, probs, boxes = self.mtcnn(frame, return_prob=True)
        return frame, probs, boxes, faces


yolo_model = Yolo()
facenet = Facenet()
