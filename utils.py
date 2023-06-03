
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import math




test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def plot_images(batch_data, batch_label, num_of_images):
    fig = plt.figure()
    cols = 4  # or however many columns you want
    rows = math.ceil(num_of_images / cols)
    for i in range(num_of_images):
        plt.subplot(rows,cols,i+1)
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()



def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train_model(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
  return train_losses, train_acc



def test_model(model, device, test_loader, criterion,test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)
            test_acc.append(100. * correct / len(target))
            test_losses.append(test_loss)


    average_test_loss = sum(test_losses)/len(test_loader.dataset)
    average_test_acc = sum(test_acc)/len(test_loader.dataset)
    

    print('Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
        average_test_loss, average_test_acc))
    
    return test_losses, test_acc


def plot_losses(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")
  plt.show()