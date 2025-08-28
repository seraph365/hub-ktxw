import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

string_labels = dataset[1].tolist()
label_to_index ={label:index for index,label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

texts = dataset[0].tolist()
# type(texts[0])

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index.update({char:len(char_to_index)})
        

class charBowDataset(Dataset):
    def __init__(self,texts,char_to_index,numerical_labels,vocab_size,max_len=40):
        super().__init__()
        self.texts = texts
        self.char_to_index = char_to_index
        self.labels = torch.tensor(numerical_labels,dtype=torch.long)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self.create_bow_vectors()
        
    def create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized_text = [self.char_to_index[char] for char in text[:self.max_len]]
            tokenized_text += [0] * (self.max_len-len(tokenized_text))
            tokenized_texts.append(tokenized_text)
            
        bow_vectors =[]
        for text in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text:
                if index != 0:
                    bow_vector[index] +=1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)
    
    def __getitem__(self, index):
        return self.bow_vectors[index],self.labels[index]
    
    def __len__(self):
        return len(self.texts)
    
    
class simpleClassfier(nn.Module):
    def __init__(self, vocab_size, hidden_dim, hidden_layers_num, output_size) -> None:
        super(simpleClassfier,self).__init__()
        self.hidden_layers_num = hidden_layers_num
        layers = []
        
        layers.append(nn.Linear(vocab_size,hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(self.hidden_layers_num):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim,output_size))
        self.model = nn.Sequential(*layers)
            
    def forward(self,x):
        out = self.model(x)
        return out
    

vocab_size = len(char_to_index)
max_len = 40
hidden_dims = [128,256]
hidden_layers_num = [1,3,5]
output_size = len(label_to_index)
num_epochs = 20
lr = 0.005

char_dataset = charBowDataset(texts, char_to_index, numerical_labels, vocab_size,max_len)
dataloader = DataLoader(char_dataset,batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for hidden_layer_num in hidden_layers_num:
    for hidden_dim in hidden_dims:
        
        model = simpleClassfier(vocab_size,hidden_dim,hidden_layer_num,output_size).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr)
        
        for epoch in range(num_epochs):
            model.train()
            runninng_loss = 0.0
            for idx,(inputs,labels) in enumerate(dataloader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                output = model(inputs)
                loss = loss_fn(output,labels)
                loss.backward()
                
                optimizer.step()
                
                runninng_loss += loss.item()
                # if idx % 50 ==0:
                #     print(f"Batch {idx}, 当前Batch loss:{loss.item()}")
            if (epoch+1) % 4 == 0:
                print(f"hidden_layer_num:{hidden_layer_num},hidden_dim:{hidden_dim},Epoch {epoch +1}/{num_epochs},loss:{runninng_loss/len(dataloader):.4f}")
                
                
            
"""Result:
Using device: cuda
hidden_layer_num:1,hidden_dim:128,Epoch 4/20,loss:2.4160
hidden_layer_num:1,hidden_dim:128,Epoch 8/20,loss:2.3540
hidden_layer_num:1,hidden_dim:128,Epoch 12/20,loss:2.2670
hidden_layer_num:1,hidden_dim:128,Epoch 16/20,loss:2.0880
hidden_layer_num:1,hidden_dim:128,Epoch 20/20,loss:1.7387
hidden_layer_num:1,hidden_dim:256,Epoch 4/20,loss:2.4198
hidden_layer_num:1,hidden_dim:256,Epoch 8/20,loss:2.3558
hidden_layer_num:1,hidden_dim:256,Epoch 12/20,loss:2.2686
hidden_layer_num:1,hidden_dim:256,Epoch 16/20,loss:2.0858
hidden_layer_num:1,hidden_dim:256,Epoch 20/20,loss:1.7114
hidden_layer_num:3,hidden_dim:128,Epoch 4/20,loss:2.4257
hidden_layer_num:3,hidden_dim:128,Epoch 8/20,loss:2.3918
hidden_layer_num:3,hidden_dim:128,Epoch 12/20,loss:2.3705
hidden_layer_num:3,hidden_dim:128,Epoch 16/20,loss:2.3588
hidden_layer_num:3,hidden_dim:128,Epoch 20/20,loss:2.3525
hidden_layer_num:3,hidden_dim:256,Epoch 4/20,loss:2.4282
hidden_layer_num:3,hidden_dim:256,Epoch 8/20,loss:2.3920
hidden_layer_num:3,hidden_dim:256,Epoch 12/20,loss:2.3702
hidden_layer_num:3,hidden_dim:256,Epoch 16/20,loss:2.3589
hidden_layer_num:3,hidden_dim:256,Epoch 20/20,loss:2.3530
hidden_layer_num:5,hidden_dim:128,Epoch 4/20,loss:2.4275
hidden_layer_num:5,hidden_dim:128,Epoch 8/20,loss:2.3914
hidden_layer_num:5,hidden_dim:128,Epoch 12/20,loss:2.3702
hidden_layer_num:5,hidden_dim:128,Epoch 16/20,loss:2.3593
hidden_layer_num:5,hidden_dim:128,Epoch 20/20,loss:2.3543
hidden_layer_num:5,hidden_dim:256,Epoch 4/20,loss:2.4323
hidden_layer_num:5,hidden_dim:256,Epoch 8/20,loss:2.3948
hidden_layer_num:5,hidden_dim:256,Epoch 12/20,loss:2.3742
hidden_layer_num:5,hidden_dim:256,Epoch 16/20,loss:2.3630
hidden_layer_num:5,hidden_dim:256,Epoch 20/20,loss:2.3555

Analysis：
1、隐藏层数量的影响:任务可能相对简单，深层模型的表达能力过剩,反而因优化困难导致学习效率低下;
2、隐藏层维度的影响:在浅层网络中，增加隐藏层维度能帮助学习更多特征，提升优化效果;

"""
