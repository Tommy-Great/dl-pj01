
import scipy.io as sio
import torch

# from torchvision.transforms import Compose,Normalize,ToTensor,ToPILImage
# import torchvision.transforms as transforms
from PIL import Image

def load_data(data_path,train_keys,dev_keys,test_keys,regularize=True,augmented=False):
    
    data_in=sio.loadmat(data_path)
    train_data,dev_data,test_data=[],[],[]
    data_list=[train_data,dev_data,test_data]
    key_list=[train_keys,dev_keys,test_keys]
    
    mean,std=0,1
    for key in train_keys:
        if 'X' in key:
            data_tmp=torch.tensor(data_in[key])
            # if augmented:
            #     data_tmp=img_augmented(data_tmp)
            data_tmp=data_tmp.float()
            mean=torch.mean(data_tmp)
            std=torch.std(data_tmp)
            data_tmp-=mean
            data_tmp/=std
        else:
            data_tmp = torch.tensor(data_in[key] % 10, dtype=torch.int64).reshape([-1])
        train_data.append(data_tmp)
    
    for item,keys in zip(data_list[1:],key_list[1:]):
        for key in keys:
            if 'X' in key:
                if regularize:
                    data_tmp = torch.tensor(data_in[key], dtype=torch.float)
                    data_tmp -= mean
                    data_tmp /= std
                    # data_tmp = data_in[key]
                    # data_tmp=img_regulize(data_tmp)
                else:
                    data_tmp=torch.tensor(data_in[key],dtype=torch.float)
            else:
                data_tmp = torch.tensor(data_in[key]%10, dtype=torch.int64).reshape([-1])
            item.append(data_tmp)

    return data_list


# def img_augmented(img_tensor):
#     tf = transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9,1.1))
#     N,D=img_tensor.shape
#     img_out = torch.zeros([2*N,D],dtype=img_tensor.dtype)
#     for i,img in enumerate(img_tensor):
#         img_tmp1=tf(img.reshape([1,16, 16]))
#         img_tmp2=tf(img.reshape([1,16, 16]))
#         img_out[i,:]=img_tmp1.reshape([16*16])
#         img_out[i+1,:]=img_tmp2.reshape([16*16])
#     img_out=torch.concat([img_tensor,img_out],dim=0)
#     return img_out


































