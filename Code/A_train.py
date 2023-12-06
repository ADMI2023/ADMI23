
from models import ts_transformer
from models import main_model
import A_dataset
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn

def SSL_train(configs):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = ts_transformer.TSTransformerEncoder(configs).to(configs.device)

    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_ssl)
    p1 = int(0.75 * configs.epoch_ssl)
    p2 = int(0.9 * configs.epoch_ssl)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )
    
    train_steps = len(train_loader)

    model.train()
    for epoch in range(configs.epoch_ssl):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (observed_data, observed_mask, observed_tp, gt_mask) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            data_batch = observed_data.to(configs.device)
            mask_batch = observed_mask.to(configs.device)
            
            loss = model.self_supervised_learning(data_batch, mask_batch)
            
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()
        lr_scheduler.step()

        if epoch % 50 == 0 or epoch == configs.epoch_ssl - 1:
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print(train_loss)

    return model

def diffusion_train(configs, pretrain_model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = main_model.CSDI_base(configs, pretrain_model).to(configs.device)

    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )

    model.train()
    for epoch in range(configs.epoch_diff):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (observed_data, observed_mask, observed_tp, gt_mask) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            loss = model(observed_data, observed_mask, observed_tp, gt_mask)
        
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch_diff-1:
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print(train_loss)
    return model

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    error_sum = 0
    missing_sum = 0

    target = []
    forecast = []
    eval_points = []

    all_generated_samples = []
    generate_sample_2d = []

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)

    start = time.time()
    for i, (observed_data, observed_mask, observed_tp, gt_mask) in enumerate(test_loader):
        imputed_samples = model.evaluate(observed_data, observed_mask, observed_tp, gt_mask).detach().to("cpu")
        imputed_sample = imputed_samples.median(dim=1).values.permute(0,2,1)

        imputed_data = observed_mask * observed_data + (1-observed_mask) * imputed_sample
        
        

        imputed_data = imputed_data.detach().to("cpu").numpy()
        observed_mask = observed_mask.detach().to("cpu").numpy()
        gt_mask = gt_mask.detach().to("cpu").numpy()
        observed_data = observed_data.detach().to("cpu").numpy()

        evalmask = gt_mask - observed_mask

        truth = observed_data * evalmask
        predict = imputed_data * evalmask
        # error = torch.sum((truth-predict)**2)
        # error_sum += error
        # missing_sum += torch.sum(evalmask)
        
        B, L, K = imputed_data.shape
        temp = imputed_data.reshape(B*L, K)
        generate_sample_2d.append(temp)

        print(str(i) + " iter test: ")

        target.append(observed_data)
        forecast.append(imputed_data)
        eval_points.append(evalmask)
        # imputed_samples = imputed_samples.permute(0, 1, 3, 2)
        # all_generated_samples.append(imputed_samples)

        end = time.time()
        print("Time:", end-start)
        start = time.time()
    
    target = np.vstack(target)
    forecast = np.vstack(forecast)
    eval_points = np.vstack(eval_points)
    # target = torch.cat(target, dim=0) #[iter, len, K]
    # forecast = torch.cat(forecast, dim=0)
    # eval_points = torch.cat(eval_points, dim=0)
    # all_generated_samples = torch.cat(all_generated_samples, dim=0)

    RMSE = calc_RMSE(target, forecast, eval_points) 
    MAE = calc_MAE(target, forecast, eval_points)
    # MAPE = calc_MAPE(target, forecast, eval_points)
    # CRPS = calc_quantile_CRPS(target, all_generated_samples, eval_points)
    print("RMSE: ", RMSE)
    print("MAE: ", MAE)
    print(RMSE)
    print(MAE)
    # print("MAPE: ", MAPE)
    # print("CRPS: ", CRPS)
    
    f = open("Z_result.txt","a")
    print("RMSE: ", RMSE, file=f)
    print("MAE: ", MAE, file=f)


    #save impute data
    generate_sample_2d = np.vstack(generate_sample_2d)
    
    # if configs.missing_rate == 0:
    np.savetxt("impute_data.csv", generate_sample_2d, delimiter=",")

def calc_RMSE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean((target[eval_p] - forecast[eval_p])**2)
    return np.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean(np.abs(target[eval_p] - forecast[eval_p]))
    return error_mean