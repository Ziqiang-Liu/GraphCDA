
import torch


def train(model, train_data, optimizer, opt):
    model.train()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score,x,y = model(train_data)
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss(score, train_data['c_d'].cuda())
        loss.backward()
        optimizer.step()
        print(loss.item())
    score = score.detach().cpu().numpy()
    return model




