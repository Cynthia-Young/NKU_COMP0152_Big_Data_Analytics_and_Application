def read_train_data(file_path):
    user_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            user_info = lines[i].strip().split('|')
            user_id = int(user_info[0])
            item_count = int(user_info[1])
            i += 1
            
            items = []
            for _ in range(item_count):
                item_info = lines[i].strip().split()
                item_id = int(item_info[0])
                score = int(item_info[1])
                items.append([item_id, score])
                i += 1
            
            user_data[user_id] = items
    return user_data

def read_test_data(file_path):
    user_items = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            user_info = lines[i].strip().split('|')
            user_id = int(user_info[0])
            item_count = int(user_info[1])
            i += 1
            
            items = []
            for _ in range(item_count):
                item_id = int(lines[i])
                items.append(item_id)
                i += 1
            
            user_items[user_id] = items
    return user_items


def read_attr_data(file_path):
    attr2id = dict()
    item2attr = dict()
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            item_info = lines[i].strip().split('|')
            if item_info[1] not in attr2id:
                attr2id[item_info[1]] = len(attr2id) + 1
            if item_info[2] not in attr2id:
                attr2id[item_info[2]] = len(attr2id) + 1
            i += 1
        
        j = 0
        while j < len(lines):
            item_info = lines[j].strip().split('|')
            item2attr[int(item_info[0])] = [attr2id[item_info[1]], attr2id[item_info[2]]]
            j += 1
    return item2attr, len(attr2id)


if __name__ == '__main__':
    file_path = 'data/test.txt'
    data_dict = read_test_data(file_path)
    print(data_dict[0])
