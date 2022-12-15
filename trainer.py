from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score 

def train(model, loader, criterion, optimizer, epoch, nb_epochs, device, writer):
    running_loss, running_corrects, running_num = 0.0, 0, 0
    y_true, y_pred = [], []
    scaler = torch.cuda.amp.GradScaler()
    tqdm_loader = tqdm(loader)
    model.train()
    for batch_idx, batch in enumerate(tqdm_loader):
        img = batch['img'].float().to(device)
        label = batch['label'].long().to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # # mixed precision + gradient clipping
        # with torch.cuda.amp.autocast():
        #     output = model(img).to(device)
        #     loss = criterion(output, label)

        # scaler.scale(loss).backward() 
        # scaler.step(optimizer)
        # scaler.update()
        
        # forward + backward + optimize
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        # stats
        preds = torch.argmax(output, dim=1)
        correct = torch.eq(preds, label).sum()
        running_loss += loss.item() 
        running_corrects += correct
        running_num += len(label)
        
        for item in label.cpu().numpy():
            y_true.append(item)
        for item in preds.cpu().numpy():
            y_pred.append(item)
        
        tqdm_loader.set_postfix({
            'Epoch': '{}/{}'.format(epoch + 1, nb_epochs),
            'Batch' : '{}/{}'.format(batch_idx + 1, len(loader)),
            'Batch Loss': '{:06f}'.format(loss.item()),
            'Mean Loss' : '{:06f}'.format(running_loss / (batch_idx + 1)),
            'Batch ACC': '{:06f}'.format(correct / len(label)),
            'Mean ACC' : '{:06f}'.format(running_corrects / running_num)
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_corrects / running_num
    epoch_f1_macro = f1_score(y_true, y_pred, average='macro')
    epoch_f1_weighted = f1_score(y_true, y_pred, average='weighted')

    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
    writer.add_scalar('F1_macro/Train', epoch_f1_macro, epoch)
    writer.add_scalar('F1_weighted/Train', epoch_f1_weighted, epoch)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
    print(f'Train loss: {epoch_loss:.6f}, Train ACC: {epoch_acc:.6f}, F1_macro: {epoch_f1_macro:.6f}, F1_weighted: {epoch_f1_weighted:.6f} lr: {optimizer.param_groups[0]["lr"]:.6f}')
    
val_loss_list, val_acc_list, val_f1_macro_list, val_f1_weighted_list = [], [], [], []

def evaluate(model, loader, criterion, epoch, device, writer, save_path):
    running_loss, running_corrects, running_num = 0.0, 0, 0
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            img = batch['img'].float().to(device)   
            label = batch['label'].long().to(device) 
            output = model(img).to(device) 
            loss = criterion(output, label)

            # stats
            preds = torch.argmax(output, dim=1)
            correct = torch.eq(preds, label).sum()
            running_loss += loss.item() 
            running_corrects += correct
            running_num += len(label)

            for item in label.cpu().numpy():
                y_true.append(item)
            for item in preds.cpu().numpy():
                y_pred.append(item)

            # tqdm_loader.set_postfix({
            #     'Epoch': '{}/{}'.format(epoch + 1, nb_epochs),
            #     'Batch' : '{}/{}'.format(batch_idx + 1, len(loader)),
            #     'Batch Loss': '{:06f}'.format(loss.item()),
            #     'Mean Loss' : '{:06f}'.format(running_loss / (batch_idx + 1)),
            #     'Batch ACC': '{:06f}'.format(correct / len(label)),
            #     'Mean ACC' : '{:06f}'.format(running_corrects / running_num)
            # })

        epoch_loss = running_loss / len(loader)
        epoch_acc = running_corrects / running_num
        epoch_f1_macro = f1_score(y_true, y_pred, average='macro')
        epoch_f1_weighted = f1_score(y_true, y_pred, average='weighted') 

        writer.add_scalar('Loss/Valid', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Valid', epoch_acc, epoch)
        writer.add_scalar('F1_macro/Valid', epoch_f1_macro, epoch)
        writer.add_scalar('F1_weighted/Valid', epoch_f1_weighted, epoch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        
        print(f'Valid loss: {epoch_loss:.6f}, Valid ACC: {epoch_acc:.6f}, F1_macro: {epoch_f1_macro:.6f}, F1_weighted: {epoch_f1_weighted:.6f}')
        print(f'{running_corrects}/{running_num} correct')
        
        epoch_acc = epoch_acc.detach().cpu().numpy()    # tensor->numpy
        val_loss_list.append(epoch_loss)
        val_acc_list.append(epoch_acc)
        val_f1_macro_list.append(epoch_f1_macro)
        val_f1_weighted_list.append(epoch_f1_weighted)

        if np.min(val_loss_list) == val_loss_list[-1]:
            # if np.max(val_f1_macro_list) == val_f1_macro_list[-1]:  # 현재 모델이 성능 최댓값이면 저장 
            torch.save(model.state_dict(), save_path) 
            print('Checkpoint Saved')

def test(model, loader, device):
    running_corrects = 0
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):
            img = batch['img'].float().to(device)
            label = batch['label'].long().to(device)

            output = model(img)
            # stats
            preds = torch.argmax(output, dim=1)
            correct = torch.eq(preds, label).sum()
            running_corrects+=correct
    
            for item in label.cpu().numpy():
                y_true.append(item)
            for item in preds.cpu().numpy():
                y_pred.append(item)

        epoch_acc = running_corrects / len(loader.dataset)
        epoch_f1_macro = f1_score(y_true, y_pred, average='macro')
        epoch_f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print(f'{running_corrects}/{len(loader.dataset)} correct')
        print(f'Test ACC: {epoch_acc:.6f}, F1_macro: {epoch_f1_macro:.6f}, F1_weighted: {epoch_f1_weighted:.6f}')

    return np.round(epoch_acc.cpu().numpy(), 4), np.round(epoch_f1_macro, 4), np.round(epoch_f1_weighted, 4)