import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import math
import datapreprocess as dp


class MLP(nn.Module):
    def __init__(self, n_user, d_user, n_item, d_item, n_attr, d_attr, out_fc1, out_fc2):
        super(MLP, self).__init__()
        self.embedding_user = nn.Embedding(n_user,d_user)
        self.embedding_item = nn.Embedding(n_item,d_item)
        self.embedding_attr = nn.Embedding(n_attr,d_attr)
        self.fc1 = nn.Linear(d_user + d_item + 2*d_attr, out_fc1)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(out_fc1, out_fc2)
        self.activate2 = nn.ReLU()
        self.fc3 = nn.Linear(out_fc2, 1)
        self.activate3 = nn.Sigmoid()
    
    def forward(self, user_ids, item_ids, attr1_ids, attr2_ids):
        vec_user = self.embedding_user(user_ids)
        vec_item = self.embedding_item(item_ids)
        vec_attr1 = self.embedding_attr(attr1_ids)
        vec_attr2 = self.embedding_attr(attr2_ids)
        x = torch.cat((vec_user, vec_item, vec_attr1, vec_attr2), 1)
        x = self.activate1(self.fc1(x))
        x = self.activate2(self.fc2(x))
        x = self.activate3(self.fc3(x))
        return x*100
    

def train(batch_size, n_epoch, lr, model_dir, train_data, val_data, item2attr, model, device):
    print('\nstart training')
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    max_mse = float('inf')

    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')
        for phase, data in [('Train', train_data), ('Valid', val_data)]:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            total_loss, total_mse = 0, 0

            for i in range(0, len(data), batch_size):
                batch = np.array(data[i:i + batch_size])
                users = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
                items = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
                ratings = torch.tensor(batch[:, 2], dtype=torch.float, device=device)
                attr1 = torch.tensor([item2attr.get(int(item), [0, 0])[0] for item in batch[:, 1]], dtype=torch.long, device=device)
                attr2 = torch.tensor([item2attr.get(int(item), [0, 0])[1] for item in batch[:, 1]], dtype=torch.long, device=device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(users, items, attr1, attr2).squeeze()
                    loss = loss_func(outputs, ratings)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                mse = mean_squared_error(ratings.cpu().numpy(), outputs.cpu().detach().numpy())
                total_loss += loss.item()
                total_mse += mse
                print(f'[{phase} Epoch {epoch+1}, Batch {i//batch_size + 1}/{len(data)//batch_size + 1}] Loss: {loss.item():.3f}, MSE: {mse:.3f}')

            print(f'\n{phase} | Loss:{total_loss/(len(data)//batch_size + 1):.5f} RMSE: {math.sqrt(total_mse/(len(data)//batch_size + 1)):.3f}')
            if phase == 'Valid' and total_mse < max_mse:
                max_mse = total_mse
                torch.save(model.state_dict(), f"{model_dir}/ckpt.model")
                print(f'Saving model with RMSE {math.sqrt(total_mse/(len(data)//batch_size + 1)):.3f}')


def process_train_data(data_dict):
    res = []
    for user_id in list(data_dict.keys()):
        values = data_dict[user_id]
        # print(values)
        for value in values:
            element = [user_id]
            element.append(value[0])
            element.append(value[1])
            res.append(element)

    return res


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file_path = 'data/train.txt'
    data_dict = dp.read_train_data(file_path)
    raw_data = process_train_data(data_dict)
    # print(raw_data)

    batch_size = 128
    epoch = 5
    lr = 0.001
    model_dir = 'result/checkpoint'
    d_user, d_item, d_attr = 100, 100, 100
    max_user, max_item, _ = np.max(raw_data, 0)
    item2attr, max_attr = dp.read_attr_data('data/itemAttribute.txt')

    model = MLP(int(max_user)+1, d_user, int(max_item)+1, d_item, max_attr+1, d_attr, 200, 100).to(device)

    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=2)

    train(batch_size, epoch, lr, model_dir, train_data, val_data, item2attr, model, device)
    
    input("please input any key to exit!")