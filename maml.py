import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import learn2learn as l2l
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from models import pHPredictionModel, ProteinpHDataset, SupportDataset
from utils import BaseTrainer, calculate_metrics, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_INTERVAL = 10

class MAMLTrainer(BaseTrainer):
    def __init__(self, model, train_loader, valid_loader, test_loader, meta_lr,
                 inner_lr, num_epochs, writer, save_dir, patience=3, min_delta=0.001,
                 validate_every=10, support_batch_size=1, inner_steps=5):
        super().__init__(model, train_loader, valid_loader, test_loader,
                        writer, save_dir, patience, min_delta, validate_every)
        
        # MAML specific initialization
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_epochs = num_epochs
        self.support_batch_size = support_batch_size
        self.inner_steps = inner_steps
        
        # Initialize MAML algorithm
        self.maml = l2l.algorithms.MAML(self.model, lr=self.inner_lr, first_order=False)
        self.optimizer = torch.optim.Adam(self.maml.parameters(), self.meta_lr)

    def adapt_on_support_set(self, learner, task_data):
        accs_spt, x_spt, y_spt = task_data['env_ids'], task_data['env_seqs'], task_data['env_pHs'].to(device)
        support_dataset = SupportDataset(accs_spt, x_spt, y_spt)
        support_loader = DataLoader(support_dataset, batch_size=self.support_batch_size, shuffle=True)
        
        for _ in range(self.inner_steps):
            total_spt_loss = 0.0
            for spt_batch in support_loader:
                spt_accs, spt_x, spt_y = spt_batch
                spt_preds = learner(spt_accs, spt_x)
                spt_loss = nn.MSELoss()(spt_preds.squeeze(), spt_y)
                total_spt_loss += spt_loss
            learner.adapt(total_spt_loss/len(support_loader), allow_nograd=True)

        return learner

    def process_task(self, task_data, train=True):
        learner = self.maml.clone()
        learner = self.adapt_on_support_set(learner, task_data)
        
        accs_qry = [task_data['opt_id']]
        x_qry = [task_data['opt_seq']]
        y_qry = task_data['opt_pH'].unsqueeze(0).to(device)
        
        qry_preds = learner(accs_qry, x_qry)
        qry_loss = nn.MSELoss()(qry_preds.squeeze(), y_qry)
        
        return qry_loss if train else (qry_loss.item(), qry_preds.item(), y_qry.item())

    def train(self):
        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                self.optimizer.zero_grad()
                meta_batch_loss = 0.0
                
                for task_data in batch:
                    qry_loss = self.process_task(task_data)
                    meta_batch_loss += qry_loss
                
                meta_batch_loss /= len(batch)
                meta_batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += meta_batch_loss.item()
                num_batches += 1
                
                # Log training progress
                avg_loss = epoch_loss / num_batches
                if global_step % LOGGING_INTERVAL == 0:
                    self.writer.add_scalar('Loss/train_moving_avg', avg_loss, global_step)
                    self.writer.flush()
                print(f'Step {global_step}, Train Loss: {avg_loss:.4f}')
                global_step += 1
                
                # Validation
                if global_step % self.validate_every == 0:
                    valid_loss = self.validate()
                    print(f'Step {global_step}, Validation Loss: {valid_loss:.4f}')
                    self.writer.add_scalar('Loss/validation', valid_loss, global_step)
                    self.writer.flush()
                    
                    if self.early_stopping(valid_loss, global_step):
                        self.model.load_state_dict(self.best_model)
                        return

            # End of epoch logging
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
                true_pHs.append(true_pH)
                predicted_pHs.append(pred_pH)
        
        metrics = calculate_metrics(true_pHs, predicted_pHs)
        print_metrics(metrics)
        
        return predictions

def get_run_name(args):
    """Generate a unique run name containing key configuration parameters"""
    pretrained_int = 1 if args.pretrained else 0
    return f"maml_{args.retrieval_strategy}_topk{args.topk}_mlr{args.meta_lr}_ilr{args.inner_lr}_ep{args.num_epochs}_ve{args.validate_every}_pr{pretrained_int}_pt{int(args.patience)}"

def main(args):
    run_name = get_run_name(args)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    args.log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Load datasets
    data_dir = os.path.join('data/processed', f'top{args.topk}', f'esm2_{args.retrieval_strategy}')
    train_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_train.json'))
    valid_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_valid.json'))
    if args.random_test:
        test_dataset = ProteinpHDataset(os.path.join('data/processed/top5/esm2_opt_random', 'retrieval_test.json'))
    else:
        test_dataset = ProteinpHDataset(os.path.join(data_dir, 'retrieval_test.json'))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=lambda x: x)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: x)
    
    # Initialize model and trainer
    model = pHPredictionModel(pretrained=args.pretrained).to(device)
    trainer = MAMLTrainer(
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
            # 只移除开头的"module."前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k  
                new_state_dict[name] = v
            trainer.model.load_state_dict(new_state_dict)
            trainer.model = trainer.model.to(device)
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
        predictions = trainer.test()
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    # Save predictions
    predictions_file = os.path.join(args.predictions_dir, f'predictions_{run_name}.csv')
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAML pH Prediction Model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--meta_lr', type=float, default=0.0001)
    parser.add_argument('--inner_lr', type=float, default=0.0005)
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
                      choices=["opt_retrieval","opt_random","opt_fixed_random"])
    parser.add_argument('--random_test', action='store_true')
    args = parser.parse_args()
    main(args)