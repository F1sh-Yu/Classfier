from train import testloader,Net,classes
import torch


if __name__ == "__main__":
    net = Net()
    net.load_state_dict(torch.load('./model.pth'))
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs,labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs,1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
            for i in range(4):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += (predicted == labels)[i].item()

    print('accuracy of the net is %f %%' %(100 * correct/total))
    for i in range(10):
        print('the accuracy of class %s : %.3f %%' %(classes[i], 100 * class_correct[i]/class_total[i]))
