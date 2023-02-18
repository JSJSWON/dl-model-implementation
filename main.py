import torch.nn as nn
import torch.optim as optim

import argparse

from load_data import *
from models import *

NUM_CLASSES = 10 # CIFAR10 dataset
IMAGE_SIZE = 224
NUM_EPOCHS = 3
LR = 0.0001
VAL_RATIO = 0.2
SHUFFLE = True
SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='train args')
    parser.add_argument('model')
    args = parser.parse_args()
    return args

models = {'alexnet': AlexNet(NUM_CLASSES)}


def train(num_epochs, train_loader, optimizer, model, loss_fn):
    for epoch in range(num_epochs):
        # train
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 500 == 499:
                print(f'epoch {epoch+1} [{500*(i//500 + 1)} / {len(train_loader)}] train loss: {train_loss}')
                train_loss = 0.0
        
        # evaluate
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
            print(f'epoch {epoch+1} val loss: {val_loss}')


if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.model


    train_loader, val_loader, test_loader = load_data(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, IMAGE_SIZE, VAL_RATIO, SHUFFLE, SEED)
    model = models[MODEL_NAME]
    model.to(DEVICE)


    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train(NUM_EPOCHS, train_loader, optimizer, model, loss_fn)
