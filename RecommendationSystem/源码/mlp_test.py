import torch
import datapreprocess as dp
import numpy as np
from mlp_train import MLP, process_train_data

def process_user_predictions(model, device, users, items, attr1, attr2, results):
    if not users:
        return
    model.eval()
    users = torch.tensor(users, dtype=torch.long).to(device)
    items = torch.tensor(items, dtype=torch.long).to(device)
    attr1 = torch.tensor(attr1, dtype=torch.long).to(device)
    attr2 = torch.tensor(attr2, dtype=torch.long).to(device)
    ratings = model(users, items, attr1, attr2).squeeze()
    results.extend(f'{items[i]}  {ratings[i].item()}\n' for i in range(len(ratings)))

if __name__ == '__main__':
    file_path = 'data/train.txt'
    data_dict = dp.read_train_data(file_path)
    raw_data = process_train_data(data_dict)
    d_user, d_item, d_attr = 100, 100, 100
    max_user, max_item, _ = np.max(raw_data, 0)
    item2attr, max_attr = dp.read_attr_data('data/itemAttribute.txt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP(int(max_user)+1, d_user, int(max_item)+1, d_item, max_attr+1, d_attr, 200, 100).to(device)
    model.load_state_dict(torch.load('result/checkpoint/ckpt.model'))
    
    results, users, items, attr1, attr2 = [], [], [], [], []

    with open('data/test.txt', 'r') as f:
        for line in f:
            if '|' in line:
                process_user_predictions(model, device, users, items, attr1, attr2, results)
                results.append(line)
                user_id = int(line.split('|')[0])
                users, items, attr1, attr2 = [], [], [], []
            else:
                item_id = int(line.strip())
                users.append(user_id)
                items.append(item_id)
                a1, a2 = item2attr.get(item_id, [0, 0])
                attr1.append(a1)
                attr2.append(a2)

    process_user_predictions(model, device, users, items, attr1, attr2, results)

    with open('result.txt', 'w') as f:
        f.writelines(results)
        
    input("please input any key to exit!")