import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as cm
from model import resnet101
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():

    def evaluate_for_test(model, model_weight_path, data_loader, device):

        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        data_loader = tqdm(data_loader, file=sys.stdout)
        results = []
        truths = []
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                images, labels = data
                pred = model(images.to(device))
                results.append(pred)
                truths.append(labels)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = os.path.join(data_root, "data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),      #train dataset
                                         transform=data_transform["train"])
    train_num = len(train_dataset)


    image_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    print("using {} spectrograms for training, {} spectrograms for validation. ".format(train_num,val_num))

    net = resnet101()
    # load pretrain weights
    model_weight_path = "./resnet101.pth"  # prein trained model
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2) # num classes
    net.to(device)


    loss_function = nn.CrossEntropyLoss()


    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 20
    best_acc = 0.0
    save_path = './best_model_Vowels.pth'   #save the well traind parameter
    train_steps = len(train_loader)


    train_loss_list=[]
    val_loss_list = []
    train_acc_list=[]
    val_acc_list=[]
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc=0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            predicts = torch.max(logits, dim=1)[1]
            train_acc+= torch.eq(predicts, labels.to(device)).sum().item()
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()


            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num
        train_loss_list.append(running_loss/train_num)
        train_acc_list.append(train_accurate)


        # validate
        net.eval()
        all_val_loss=0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_loss = loss_function(outputs, val_labels.to(device))
                all_val_loss+=val_loss.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)


        val_accurate = acc / val_num
        val_loss=loss_function(outputs, val_labels.to(device))
        val_loss_list.append(all_val_loss/val_num)
        val_acc_list.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print('save model')



    print('Finished Training')
    val_model_weight_path = './best_model.pth'
    val_model = resnet101()
    val_model.fc = nn.Linear(in_channel, 2)
    val_model.to(device)
    results, truths = evaluate_for_test(val_model, val_model_weight_path, validate_loader, device=device)

    results = torch.argmax(results, dim=1)
    truths = truths.cpu().numpy()
    results = results.cpu().numpy()
    matrix = cm(truths, results, labels=[0, 1])

    confusion_labels = ['unvoiced', 'voiced']
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=confusion_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./confusion.jpg')
    plt.show()


    #############################
    x = list(range(1, len(train_loss_list)+1))
    y=list(range(1,len(train_acc_list)+1))
    plt.figure()
    plt.title("train/test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, train_loss_list, label="train loss")
    plt.plot(x, val_loss_list, label="test loss")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('./loss.jpg')

    plt.figure()
    plt.title("train/test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(y, train_acc_list, label='train accuracy')
    plt.plot(y, val_acc_list, label='test accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('./accuracy.jpg')
    plt.show()



if __name__ == '__main__':
    main()

