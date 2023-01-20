import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image


# Creating a custom dataset to store images and corresponding labels
class MyDataset(Dataset):
    def __init__(self, task, part, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        # Read the labels from the Excel file
        if (task == 'A' and part == 0): 
            self.labels = pd.read_csv('./Datasets/celeba/labels.csv', usecols=[2])['gender'].tolist()
        if (task == 'A' and part == 1): 
            self.labels = pd.read_csv('./Datasets/celeba_test/labels.csv', usecols=[2], delimiter='\t')['gender'].tolist()
        if (task == 'A' and part == 2):
            self.labels = pd.read_csv('./Datasets/celeba_test/labels.csv', usecols=[3], delimiter='\t')['smiling'].tolist()
        if (task == 'B' and part == 1): 
            self.labels = pd.read_csv('./Datasets/cartoon_set_test/labels.csv', usecols=[2])['face_shape'].tolist()
        if (task == 'B' and part == 2):
            self.labels = pd.read_csv('./Datasets/cartoon_set_test/labels.csv', usecols=[1])['eye_color'].tolist()
        
        # Get the list of image files (without the lambda it was not in order)
        self.image_files = []
        for root, dirs, files in os.walk(root_dir): 
            self.image_files.extend([os.path.join(root, file) for file in sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image and apply the transformation
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Convert the label to a tensor
        label = torch.tensor(self.labels[idx])
        
        return image, label



def main():
    # Setting random seed for reproducability
    torch.manual_seed(42)

    # Defining the transforms from the paper (train needed for A1 as Regnet model is too large)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        # Requirements for pretrained models:
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
            # Requirements for pretrained models:
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    # Define the root directory of the dataset
    root_dir_trainA = './Datasets/celeba/img'
    root_dir_testA = './Datasets/celeba_test/img'
    root_dir_testB = './Datasets/cartoon_set_test/img'

    # Define the class names and their corresponding indexes / this is not used here and is only included for simplicity
    class_names = ['man', 'woman']
    class_to_idx = {'man': 1, 'woman': -1}
    class_names2 = ['Shape0', 'Shape1', 'Shape2', 'Shape3', 'Shape4']
    class_to_idx2 = {'Shape0': 0, 'Shape1': 1, 'Shape2': 2, 'Shape3': 3, 'Shape4': 4}


    A1train_data = MyDataset('A', 0, root_dir_trainA, transform=train_transform)
    A1test_data = MyDataset('A', 1, root_dir_testA, transform=test_transform)
    A2test_data = MyDataset('A', 2, root_dir_testA, transform=test_transform)
    B1test_data = MyDataset('B', 1, root_dir_testB, transform=test_transform)
    B2test_data = MyDataset('B', 2, root_dir_testB, transform=test_transform)

    A1train_loader = DataLoader(A1test_data, batch_size=64, shuffle=True) # Regnet was trained on 64 batches due to memory issues on my PC
    A1test_loader = DataLoader(A1test_data, batch_size=125, shuffle=False)
    A2test_loader = DataLoader(A2test_data, batch_size=125, shuffle=False)
    B1test_loader = DataLoader(B1test_data, batch_size=125, shuffle=False)
    B2test_loader = DataLoader(B2test_data, batch_size=125, shuffle=False)

    ##############################################################################################################
    #                                                                                                            #
    #                                                  A1                                                        #
    #                                                                                                            #
    ##############################################################################################################

    # Defining the model
    RegnetY16GF = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    
    # Freezing parameters
    for param in RegnetY16GF.parameters():
        param.requires_grad = False
    
    # modifying classifier
    torch.manual_seed(42)

    RegnetY16GF.fc = nn.Sequential(nn.Linear(3024,1000),nn.Linear(1000,2),nn.LogSoftmax(dim=1))

    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(RegnetY16GF.fc.parameters(), lr=0.001)

    # Unfreezing parameters for fc
    for param in RegnetY16GF.fc.parameters():
        param.requires_grad = True

    # Training the RegnetY16GF model as I could not upload it to github due to it being too large
    epochs = 6

    max_trn_batch = 78 # 78 * 64 = 4992
    max_tst_batch = 15 # 15 * 64 = 960

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
            
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            
            # Changing the labels to be in the range [0, num_classes-1].
            y_train = torch.where(y_train == 1, torch.tensor(1), torch.tensor(0))


            if b == max_trn_batch:
                break
            b+=1
            
            # Apply the model
            y_pred = RegnetY16GF(X_train)
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b%(max_trn_batch/2) == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{64*b:4}/4992]  loss: {loss.item():10.8f}  \  accuracy: {trn_corr.item()*100/(64*b):7.2f}%')

    A1actual = []
    A1predictions = []

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(A1test_loader):
            # Changing the labels to be in the range [0, num_classes-1].
            y_test = torch.where(y_test == 1, torch.tensor(1), torch.tensor(0))

            # Apply the model
            y_val = A1model(X_test)

            # Predictions
            predicted = torch.max(y_val.data, 1)[1]
            A1predictions.append(predicted)
            A1actual.append(y_test)

    #Converting tensors to list
    A1preds = []
    for tensor in A1predictions:
        A1preds.append(tensor.tolist())

    A1act = []
    for tensor in A1actual:
        A1act.append(tensor.tolist())

    A1act = np.reshape(np.array(A1act), (1000,))
    A1preds = np.reshape(np.array(A1preds), (1000,))

    ##############################################################################################################
    #                                                                                                            #
    #                                                  A2                                                        #
    #                                                                                                            #
    ##############################################################################################################

    # Defining the model
    class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # (((178−2)/2)−2)/2=43
        # (((218−2)/2)−2)/2=53
        self.fc1 = nn.Linear(43*53*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 43*53*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

    A2model = ConvolutionalNetwork()
    A2model.load_state_dict(torch.load('./A2/A2CNNModelFinal.pt'))

    A2actual = []
    A2predictions = []

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(A2test_loader):
            # Changing the labels to be in the range [0, num_classes-1].
            y_test = torch.where(y_test == 1, torch.tensor(1), torch.tensor(0))

            # Apply the model
            y_val = A2model(X_test)

            # Predictions
            predicted = torch.max(y_val.data, 1)[1]
            A2predictions.append(predicted)
            A2actual.append(y_test)

    #Converting tensors to list
    A2preds = []
    for tensor in A2predictions:
        A2preds.append(tensor.tolist())

    A2act = []
    for tensor in A2actual:
        A2act.append(tensor.tolist())


    A2act = np.reshape(np.array(A2act), (1000,))
    A2preds = np.reshape(np.array(A2preds), (1000,))

    ##############################################################################################################
    #                                                                                                            #
    #                                                  B1                                                        #
    #                                                                                                            #
    ##############################################################################################################

    # Defining the model
    class ConvolutionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3, 1)
            self.conv2 = nn.Conv2d(6, 16, 3, 1)
            # (((500−2)/2)−2)/2=123.5
            # (((500−2)/2)−2)/2=123.5
            self.fc1 = nn.Linear(123*123*16, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 5)

        def forward(self, X):
            X = F.relu(self.conv1(X))
            X = F.max_pool2d(X, 2, 2)
            X = F.relu(self.conv2(X))
            X = F.max_pool2d(X, 2, 2)
            X = X.view(-1, 123*123*16)
            X = F.relu(self.fc1(X))
            X = F.relu(self.fc2(X))
            X = self.fc3(X)
            return F.log_softmax(X, dim=1)

    B1model = ConvolutionalNetwork()
    B1model.load_state_dict(torch.load('./B2/B2CNNModelFinal.pt')) # B2 is called as I accidentally mixed up B1 and B2

    # Applying the model to the test set
    B1actual = []
    B1predictions = []

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(B1test_loader):

            # Apply the model
            y_val = B1model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            B1predictions.append(predicted)
            B1actual.append(y_test)

    #Converting tensors to list
    B1preds = []
    for tensor in B1predictions:
        B1preds.append(tensor.tolist())

    B1act = []
    for tensor in B1actual:
        B1act.append(tensor.tolist())

    B1act = np.reshape(np.array(B1act), (2500,))
    B1preds = np.reshape(np.array(B1preds), (2500,))

    ##############################################################################################################
    #                                                                                                            #
    #                                                  B2                                                        #
    #                                                                                                            #
    ##############################################################################################################

    # Defining the model
    class ConvolutionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3, 1)
            self.conv2 = nn.Conv2d(6, 16, 3, 1)
            # (((500−2)/2)−2)/2=123.5
            # (((500−2)/2)−2)/2=123.5
            self.fc1 = nn.Linear(123*123*16, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 5)

        def forward(self, X):
            X = F.relu(self.conv1(X))
            X = F.max_pool2d(X, 2, 2)
            X = F.relu(self.conv2(X))
            X = F.max_pool2d(X, 2, 2)
            X = X.view(-1, 123*123*16)
            X = F.relu(self.fc1(X))
            X = F.relu(self.fc2(X))
            X = self.fc3(X)
            return F.log_softmax(X, dim=1)

    B2model = ConvolutionalNetwork()
    B2model.load_state_dict(torch.load('./B1/B2CNNModelFinal.pt'))

    # Applying model on testset
    B2actual = []
    B2predictions = []

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(B2test_loader):
            # Apply the model
            y_val = B2model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            B2predictions.append(predicted)
            B2actual.append(y_test)

    #Converting tensors to list
    B2preds = []
    for tensor in B2predictions:
        B2preds.append(tensor.tolist())

    B2act = []
    for tensor in B2actual:
        B2act.append(tensor.tolist())

    B2act = np.reshape(np.array(B2act), (2500,))
    B2preds = np.reshape(np.array(B2preds), (2500,))

    ##############################################################################################################
    #                                                                                                            #
    #                                                Output                                                      #
    #                                                                                                            #
    ##############################################################################################################

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))

    # Create the first confusion matrix
    conf_matrix1 = confusion_matrix(y_true=A1act, y_pred=A1preds)
    axs[0, 0].matshow(conf_matrix1, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix1.shape[0]):
        for j in range(conf_matrix1.shape[1]):
            axs[0, 0].text(x=j, y=i, s=conf_matrix1[i, j], va='center', ha='center', size='xx-large')
    axs[0, 0].set_xlabel('Predictions', fontsize=18)
    axs[0, 0].set_ylabel('Actuals', fontsize=18)
    axs[0, 0].set_title('Confusion Matrix A1', fontsize=18)

    # Create the second confusion matrix
    conf_matrix2 = confusion_matrix(y_true=A2act, y_pred=A2preds)
    axs[0, 1].matshow(conf_matrix2, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix2.shape[0]):
        for j in range(conf_matrix2.shape[1]):
            axs[0, 1].text(x=j, y=i, s=conf_matrix2[i, j], va='center', ha='center', size='xx-large')
    axs[0, 1].set_xlabel('Predictions', fontsize=18)
    axs[0, 1].set_ylabel('Actuals', fontsize=18)
    axs[0, 1].set_title('Confusion Matrix A2', fontsize=18)

    # Create the third confusion matrix
    conf_matrix3 = confusion_matrix(y_true=B1act, y_pred=B1preds)
    axs[1, 0].matshow(conf_matrix3, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix3.shape[0]):
        for j in range(conf_matrix3.shape[1]):
            axs[1, 0].text(x=j, y=i, s=conf_matrix3[i, j], va='center', ha='center', size='xx-large')
    axs[1, 0].set_xlabel('Predictions', fontsize=18)
    axs[1, 0].set_ylabel('Actuals', fontsize=18)
    axs[1, 0].set_title('Confusion Matrix B1', fontsize=18)

    # Create the fourth confusion matrix
    conf_matrix4 = confusion_matrix(y_true=B2act, y_pred=B2preds)
    axs[1, 0].matshow(conf_matrix4, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix4.shape[0]):
        for j in range(conf_matrix4.shape[1]):
            axs[1, 0].text(x=j, y=i, s=conf_matrix4[i, j], va='center', ha='center', size='xx-large')
    axs[1, 0].set_xlabel('Predictions', fontsize=18)
    axs[1, 0].set_ylabel('Actuals', fontsize=18)
    axs[1, 0].set_title('Confusion Matrix B2', fontsize=18)

    plt.show()




if __name__ == "__main__":
    main()
