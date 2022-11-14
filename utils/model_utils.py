import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
def train_torch_model(train_loader, model, valid_loader, epoch_number,
                      model_path, learning_rate, predict_length, model_name, weight_decay=0.005):
    optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=weight_decay, params=model.parameters())
    loss_func = nn.MSELoss()
    loss_list = []

    valid_list = [1.0e20]
    print("[Info] We start training the model!")
    for epoch in range(epoch_number):
        total_loss_value = 0.0
        for idx, (start_token, predict_token, batch_item, batch_app, batch_zone) in enumerate(train_loader):
            optimizer.zero_grad()

            embedding_token = torch.cat([batch_item, batch_app, batch_zone], dim=-1)
            dec_token_seq = torch.zeros_like(start_token)[:, :predict_length, :]
            dec_token = torch.cat([start_token, dec_token_seq], dim=1)

            output_tensor = model(x_enc=start_token, x_mark_enc=embedding_token, x_dec=dec_token)

            loss_value = loss_func(output_tensor, predict_token)

            loss_value.backward()
            optimizer.step()

            total_loss_value = total_loss_value + loss_value.item()
        loss_list.append(total_loss_value)
        print(f"[Info] Epoch: {epoch+1}, loss: {round(loss_list[-1], 6)}")
        # 绷不住了搞这个
        torch.save(model.state_dict(), model_path + "/" + model_name)
        print("[Info] We saved the trained model!")
        if (epoch + 1) % 5 == 0:
            # 进行验证
            model.eval()
            with torch.no_grad():
                total_loss_value = 0.0
                for idx, (start_token, predict_token, batch_item, batch_app, batch_zone) in enumerate(valid_loader):

                    embedding_token = torch.cat([batch_item, batch_app, batch_zone], dim=-1)
                    dec_token_seq = torch.zeros_like(start_token)[:, :predict_length, :]
                    dec_token = torch.cat([start_token, dec_token_seq], dim=1)
                    output_tensor = model(x_enc=start_token, x_mark_enc=embedding_token, x_dec=dec_token)

                    loss_value = loss_func(output_tensor, predict_token)
                    total_loss_value = total_loss_value + loss_value.item()
                if total_loss_value < valid_list[-1]:
                    valid_list.append(total_loss_value)
                    torch.save(model.state_dict(), model_path + "/" + model_name)
                print(f"[Info] Validation loss: {round(total_loss_value, 6)}")


class inferenceDataset(Dataset):
    def __init__(self, start_token_tensor, item_tensor, app_tensor, zone_tensor):
        self.start_token = start_token_tensor
        self.item_tensor = item_tensor
        self.app_tensor = app_tensor
        self.zone_tensor = zone_tensor

    def __getitem__(self, index):
        return self.start_token[index, :, :], self.item_tensor[index, :], self.app_tensor[index, :], self.zone_tensor[index, :]

    def __len__(self):
        return self.start_token.shape[0]

def inference_model(model, start_token_tensor, item_tensor, app_tensor, zone_tensor, batch_size, predict_length):
    """
    :param model:
    :param start_token_list:
    :param batch_item:
    :param batch_app:
    :param batch_zone:
    :return:
    """
    # 推理数据
    part_dataset = inferenceDataset(start_token_tensor=start_token_tensor,
                                    item_tensor=item_tensor, app_tensor=app_tensor,
                                    zone_tensor=zone_tensor)
    part_dataloader = DataLoader(dataset=part_dataset, batch_size=batch_size, shuffle=False)
    result_list = []
    model.eval()
    with torch.no_grad():
        for idx, (start_token, batch_item, batch_app, batch_zone) in enumerate(part_dataloader):

            print(batch_item.shape, batch_app.shape, batch_zone.shape)

            embedding_token = torch.cat([batch_item, batch_app, batch_zone], dim=-1)
            dec_token_seq = torch.zeros_like(start_token)[:, :predict_length, :]
            dec_token = torch.cat([start_token, dec_token_seq], dim=1)
            output_tensor = model(x_enc=start_token, x_mark_enc=embedding_token, x_dec=dec_token)
            result_list.append(output_tensor.cpu())

    total_result_tensor = torch.cat(result_list, dim=0)
    return total_result_tensor

