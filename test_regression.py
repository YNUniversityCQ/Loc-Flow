import time
import torch
from tqdm import tqdm
from lib import loader, modules
from torch.utils.data import Dataset, DataLoader

# loading model
device = torch.device("cuda:0")
model = modules.Loc_Flow()
model.load_state_dict(torch.load('results/DPMBestModel.pt'))
model.to(device)

def my_loss(output, target):
    output = output.pred_jts
    loss = torch.sum((output * 255 - target * 255) ** 2, 1)
    loss = torch.sqrt(loss)
    loss = torch.mean(loss)
    return loss
def main_worker():

    # loading test data
    test_data = loader.locDL(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    loss = []

    start_time = time.time()
    for inputs, loc in tqdm(test_dataloader):
        interation += 1

        inputs, target = inputs.cuda(), loc.cuda()

        with torch.no_grad():
            pre = model(inputs)
            # print(pre, target)
            losses = my_loss(pre, target)
            # print(losses)

        loss.append(losses)

    end_time = time.time()
    runtime = end_time - start_time
    print("Total runtime: {:.2f} seconds".format(runtime))

    rmse_err = sum(loss)/len(loss)

    print('测试集平均绝对误差：', rmse_err)

if __name__ == '__main__':
 main_worker()