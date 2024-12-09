import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from models import pHPredictionModel, ProteinpHDataset, SupportDataset
from utils import BaseTrainer, calculate_metrics, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_INTERVAL = 10
    
class ReptileTrainer(BaseTrainer):
    def __init__(self, model, train_loader, valid_loader, test_loader, meta_lr,
                 inner_lr, num_epochs, writer, save_dir, patience=3, min_delta=0.001,
                 validate_every=10, support_batch_size=1, inner_steps=5):
        super().__init__(model, train_loader, valid_loader, test_loader,
                        writer, save_dir, patience, min_delta, validate_every)
        
        # Reptile specific initialization
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_epochs = num_epochs
        self.support_batch_size = support_batch_size
        self.inner_steps = inner_steps
        
        self.inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

    def adapt_on_support_set(self, task_data):
        """执行内循环适应"""
        # 保存原始参数
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # 一次性处理所有support set数据
        accs_spt, x_spt, y_spt = task_data['env_ids'], task_data['env_seqs'], task_data['env_pHs'].to(device)
        support_dataset = SupportDataset(accs_spt, x_spt, y_spt)
        support_loader = DataLoader(support_dataset, batch_size=self.support_batch_size, shuffle=True)
        
        for _ in range(self.inner_steps):
            total_spt_loss = 0.0
            for spt_batch in support_loader:
                spt_accs, spt_x, spt_y = spt_batch
                spt_preds = self.model(spt_accs, spt_x)
                loss = nn.MSELoss()(spt_preds.squeeze(), spt_y)
                total_spt_loss += loss
            avg_spt_loss = total_spt_loss / len(support_loader)
            self.inner_optimizer.zero_grad()
            avg_spt_loss.backward()
            self.inner_optimizer.step()
        # 获取适应后的参数
        adapted_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # 恢复原始参数
        self.model.load_state_dict(original_state)
        
        return adapted_state


    def process_task(self, task_data, train=True):
        if train:
            # 训练模式：返回适应后的参数
            return self.adapt_on_support_set(task_data)
        else:
            # 评估模式：计算损失和预测值
            adapted_state = self.adapt_on_support_set(task_data)
            
            # adapt_on_support_set中已将模型参数还原为原始参数
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            #使用适应后的参数进行预测
            self.model.load_state_dict(adapted_state)
            
            accs_qry = [task_data['opt_id']]
            x_qry = [task_data['opt_seq']]
            y_qry = task_data['opt_pH'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                qry_preds = self.model(accs_qry, x_qry)
                qry_loss = nn.MSELoss()(qry_preds.squeeze(), y_qry)
            
            # 恢复原始参数
            self.model.load_state_dict(original_state)
            
            return qry_loss.item(), qry_preds.item(), y_qry.item()
        
    
                    
    def train(self):
        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                meta_batch_loss = 0.0
                # 适应每个任务
                for task_data in batch:
                    adapted_state = self.process_task(task_data)
                    # 计算并应用元梯度更新
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                grad = (adapted_state[name] - param) * self.meta_lr
                                param.data.add_(grad)
                        qry_preds = self.model([task_data['opt_id']], [task_data['opt_seq']])
                        qry_loss = nn.MSELoss()(qry_preds.squeeze(), task_data['opt_pH'].to(device))
                        meta_batch_loss += qry_loss.item()
                # 计算平均损失
                meta_batch_loss /= len(batch)
                epoch_loss += meta_batch_loss
                num_batches += 1
                
                # 记录训练进度
                avg_loss = epoch_loss / num_batches
                if global_step % LOGGING_INTERVAL == 0:
                    self.writer.add_scalar('Loss/train_moving_avg', avg_loss, global_step)
                    self.writer.flush()
                print(f'Step {global_step}, Train Loss: {avg_loss:.4f}')
                global_step += 1
                
                # 验证
                if global_step % self.validate_every == 0:
                    valid_loss = self.validate()
                    print(f'Step {global_step}, Validation Loss: {valid_loss:.4f}')
                    self.writer.add_scalar('Loss/validation', valid_loss, global_step)
                    self.writer.flush()
                    
                    if self.early_stopping(valid_loss, global_step):
                        self.model.load_state_dict(self.best_model)
                        return
            
            # 每个epoch结束时的记录
            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch {epoch+1}, Average Train Loss: {avg_epoch_loss:.4f}')
            self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)

    def validate(self):
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.valid_loader, desc="Validation"):
            for task_data in batch:
                qry_loss, _, _ = self.process_task(task_data, train=False)
                total_loss += qry_loss
            num_batches += 1
        return total_loss / num_batches

    def test(self):
        predictions = []
        true_pHs = []
        predicted_pHs = []
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            for task_data in batch:
                _, pred_pH, true_pH = self.process_task(task_data, train=False)
                predictions.append({
                    'opt_id': task_data['opt_id'],
                    'true_pH': true_pH,
                    'predicted_pH': pred_pH
                })
                # print(predictions)
                true_pHs.append(true_pH)
                predicted_pHs.append(pred_pH)
        
        metrics = calculate_metrics(true_pHs, predicted_pHs)
        print_metrics(metrics)
        
        return predictions

def get_run_name(args):
    """生成包含关键配置参数的唯一运行名称"""
    pretrained_int = 1 if args.pretrained else 0
    return f"reptile_{args.retrieval_strategy}_topk{args.topk}_mlr{args.meta_lr}_ilr{args.inner_lr}_ep{args.num_epochs}_ve{args.validate_every}_pr{pretrained_int}_pt{int(args.patience)}"

def main(args):
    run_name = get_run_name(args)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    args.log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 加载数据集
    data_dir = os.path.join('data/processed', f'top{args.topk}', f'esm2_{args.retrieval_strategy}')
    train_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_train.json'))
    valid_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_valid.json'))
    if args.random_test:
        test_dataset = ProteinpHDataset(os.path.join('data/processed/top5/esm2_opt_random', 'retrieval_test.json'))
    else:
        test_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_test.json'))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, 
                            collate_fn=lambda x: x, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False, 
                            collate_fn=lambda x: x, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, 
                           collate_fn=lambda x: x, num_workers=4, pin_memory=True)
    
    # 初始化模型和训练器
    model = pHPredictionModel(pretrained=args.pretrained).to(device)
    trainer = ReptileTrainer(
        model, train_loader, valid_loader, test_loader,
        args.meta_lr, args.inner_lr, args.num_epochs,
        writer, args.save_dir, patience=args.patience,
        min_delta=args.min_delta, validate_every=args.validate_every,
        support_batch_size=args.support_batch_size,
        inner_steps=args.inner_steps
    )

    if args.mode == 'train':
        trainer.train()
        predictions = trainer.test()
    elif args.mode == 'test':
        model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            # print(state_dict)
            trainer.model.load_state_dict(state_dict['model_state_dict'])
            trainer.model = trainer.model.to(device)
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
        predictions = trainer.test()
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    # 保存预测结果
    predictions_file = os.path.join(args.predictions_dir, f'predictions_{run_name}.csv')
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reptile pH Prediction Model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--meta_lr', type=float, default=1)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--predictions_dir', type=str, default='./predictions')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--support_batch_size', type=int, default=1)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--validate_every', type=int, default=200)
    parser.add_argument("--retrieval_strategy", default="opt_retrieval",
                      choices=["opt_retrieval","opt_random","opt_fixed_random",
                               "opt_retrieval_scaled_0.2", 
                               "opt_retrieval_scaled_0.6"])
    parser.add_argument('--random_test', action='store_true')
    
    args = parser.parse_args()
    main(args)