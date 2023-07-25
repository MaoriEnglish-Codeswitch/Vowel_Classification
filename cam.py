import numpy as np
import cv2
import argparse
import torch
from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk
from model import resnet101
import os

#function definitions
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #print(cam_img.shape)
        #print(cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        #print(output_cam)
    return cam_img, output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, all_classes[class_idx[i]], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('CAM', result/255.)
        print(save_name)
        cv2.waitKey(0)
        cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)

def load_synset_classes(file_path):
    # load the synset text file for labels
    all_classes = []
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        labels = [line.split('\n') for line in all_lines]
        for label_list in labels:
            current_class = [name.split(',') for name in label_list][0][0][10:]
            all_classes.append(current_class)
    return all_classes

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# load the model

# model = models.resnet18(pretrained=True).eval()
#################

model_path='./Model_resnet101_4000_voicedunvoiced/unv_v_20_epochs/best_model_unv_and_v.pth'
model = resnet101(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
##############################
# hook the feature extractor
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []


model._modules.get('layer4').register_forward_hook(hook_feature)
# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

# define the transforms, resize => tensor => normalize
transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

dir='./test3/'
all_vowels=os.listdir(dir)
size_upsample = (256, 256)
CAM_array_np = np.zeros((7,7))
#print(CAM_array)
output_cam1 = []
for each_vowel in all_vowels:
    #print(all_vowels)
    CAM_array = []
    each_vowel_dir = os.path.join(dir, each_vowel)
    #print(each_vowel_dir)
    all_spectrogram = os.listdir(each_vowel_dir)
    print(all_spectrogram)
    for each_spectrogram in all_spectrogram:
        print(each_spectrogram.split(".")[1])
        if (each_spectrogram.split(".")[1] == "jpg"):
            each_spectrogram_dir = os.path.join(each_vowel_dir, each_spectrogram)
            # construct the argument parser
            parser = argparse.ArgumentParser()
            parser.add_argument('-i', '--input', help='path to input image', default=each_spectrogram_dir)
            args = vars(parser.parse_args())
            # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py

            # get all the classes in a list
            all_classes = load_synset_classes('Voiced_Unvoiced.txt')

            # read and visualize the image
            image = cv2.imread(args['input'])
            orig_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape


            # apply the image transforms
            image_tensor = transforms(image)
            # add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            # forward pass through model
            outputs = model(image_tensor)
            # get the softmax probabilities
            probs = F.softmax(outputs).data.squeeze()
            # get the class indices of top k probabilities
            class_idx = topk(probs, 1)[1].int()

            # generate class activation mapping for the top1 prediction
            CAM_np, CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)

            CAM_array_np = np.mean(np.array([CAM_np, CAM_array_np]), axis=0)
            #CAM_array_np = np.mean(CAM_np, CAM_array_np)
            #print(CAM_array_np)
            # file name to save the resulting CAM image with
            save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
            save_name = save_name.split("L")[1]
                    # show and save the results
            #show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name)
            for i, cam in enumerate(CAMs):
                heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
                img_gray = cv2.applyColorMap(orig_image, cv2.COLORMAP_BONE)
                result = heatmap * 1 + orig_image * 0
                # put class label text on the result
                cv2.putText(result, all_classes[class_idx[i]], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                #cv2.imshow('CAM', result / 255.)
               # print(save_name)
                #cv2.waitKey(0)
                cv2.imwrite(f"CAM_voicedunvoiced_heatmap_{save_name}.jpg", result)
    #CAM_array = np.array(CAM_array)
   # print(np.average(CAM_array, axis= 1))
    #output_cam1.append(cv2.resize(CAM_array_np, size_upsample))
    #save_name_vowel = f"{args['input'].split('/')[-2].split('.')[0]}"
   # print(save_name_vowel)
   # show_cam(output_cam1, width, height, orig_image, class_idx, all_classes, save_name_vowel)