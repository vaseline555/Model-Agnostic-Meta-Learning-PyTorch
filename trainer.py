import torch
import copy

import scipy.stats
import numpy as np

from collections import OrderedDict
from tqdm import trange
from torch import nn


from utils import initiate_model, set_weights, compute_adapted_weights, update_model_weights, zero_grad

class MAMLTrainer:
    """Trainer module for MAML
    """
    def __init__(self, args, model, device):
        """Initiate MAML trainer
        
        Args:
            model: model instance used for meta-learning
            args: arguments entered by the user
        """
        self.N = args.n_way
        self.K = args.k_spt
        self.Q = args.k_qry
        self.B = args.meta_batch_size
        self.meta_train_step = args.meta_train_step
        self.meta_test_step = args.meta_test_step
        self.num_test_step = args.num_test_points // self.B
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.device = device
        self.model = initiate_model(model, args.n_way, self.device)
    
    def train_step(self, dataloader):
        """Method for meta-training
        
        Args:
            dataloader: torch.utils.data.DataLoader object to sample support & query sets from
        
        Returns:
            te_losses_total: average losses on query set
            te_losses_ci: 95% confidence interval lof losses on query set
            te_accs_total: average accuracies on query set
            te_accs_ci: 95% confidence interval lof accuracies on query set
        """
        # define model for meta training
        weights = OrderedDict((name, param) for (name, param) in self.model.named_parameters())
            
        # iterative updates
        te_losses, te_accs = 0.0, 0.0
        for _ in trange(self.meta_train_step, desc='[INFO] Meta-training', leave=False):
            meta_gradients = []
            for _ in range(self.B):           
                # get dataset
                Sx, Sy, Qx, Qy = next(iter(dataloader))
                Sx, Sy = Sx.to(self.device).view(self.N * self.K, *Sx.shape[2:]), Sy.to(self.device).view(-1, 1).squeeze()
                Qx, Qy = Qx.to(self.device).view(self.N * self.Q, *Qx.shape[2:]), Qy.to(self.device).view(-1, 1).squeeze()

                # inference & get loss
                Sy_hat = self.model(Sx)
                tr_loss = self.criterion(Sy_hat, Sy)

                # get support gradient & hessian
                zero_grad(self.model.parameters())
                support_grad = torch.autograd.grad(tr_loss, self.model.parameters(), create_graph=True)

                # calculate updated weights (theta_prime)
                adapted_weights = compute_adapted_weights(weights, support_grad, self.alpha)

                # get loss on adapted parameter to new tasks (with computational graph attached)
                Qy_hat = self.model(Qx, adapted_weights)
                te_loss = self.criterion(Qy_hat, Qy)

                # get query gradient
                zero_grad(self.model.parameters())
                query_grad = torch.autograd.grad(te_loss, self.model.parameters())

                # save for meta-gradient
                meta_gradients.append(query_grad)
            else:
                # calculate sum of all meta gradients
                meta_gradient_final = tuple(map(sum, zip(*meta_gradients)))

                # calculate updated weights (theta)
                weights = compute_adapted_weights(weights, meta_gradient_final, self.beta)

                # assign weights to the original model
                self.model = update_model_weights(self.model, weights)
                
                # clear gradient cache
                del support_grad, query_grad
        else:
            # get metric and loss
            te_losses += te_loss.item()

            # get correct counts
            predicted = Qy_hat.argmax(dim=1, keepdim=True)
            te_accs += predicted.eq(Qy.view_as(predicted)).sum().item()
            
             # update metrics for epoch
            te_losses /= self.N * self.Q
            te_accs /= self.N * self.Q
        return te_losses, te_accs

    def eval_step(self, dataloader):
        """Method for meta-testing
        
        Args:
            dataloader: torch.utils.data.DataLoader object to sample support & query sets from
            mode: 'val'/'test'
                * 'val': evaluate on valiation set (same as `train_step` EXCEPT outer loop)
                * 'test': evaluate on test set (test on randomly sampled `args.num_test_points` episodes)
        
        Returns:
            te_losses_total: average losses on query set
            te_losses_ci: 95% confidence interval lof losses on query set
            te_accs_total: average accuracies on query set
            te_accs_ci: 95% confidence interval lof accuracies on query set
        """     
        # define new model for meta-test
        meta_tester = copy.deepcopy(self.model)
            
        # define model for meta training
        weights = OrderedDict((name, param) for (name, param) in meta_tester.named_parameters())
            
        # iterative updates
        te_losses, te_accs = 0.0, 0.0
        for _ in trange(self.meta_test_step, desc='[INFO] Meta-training', leave=False):
            for _ in range(self.B):           
                # get dataset
                Sx, Sy, Qx, Qy = next(iter(dataloader))
                Sx, Sy = Sx.to(self.device).view(self.N * self.K, *Sx.shape[2:]), Sy.to(self.device).view(-1, 1).squeeze()
                Qx, Qy = Qx.to(self.device).view(self.N * self.Q, *Qx.shape[2:]), Qy.to(self.device).view(-1, 1).squeeze()

                # inference & get loss
                Sy_hat = self.model(Sx)
                tr_loss = self.criterion(Sy_hat, Sy)

                # get support gradient & hessian
                zero_grad(self.model.parameters())
                support_grad = torch.autograd.grad(tr_loss, self.model.parameters(), create_graph=True)

                # calculate updated weights (theta_prime)
                adapted_weights = compute_adapted_weights(weights, support_grad, self.alpha)

                # get loss on adapted parameter to new tasks (with computational graph attached)
                Qy_hat = self.model(Qx, adapted_weights)
                te_loss = self.criterion(Qy_hat, Qy)
            else:
                # clear gradient cache
                del support_grad
        else:
            # get metric and loss
            te_losses += te_loss.item()

            # get correct counts
            predicted = Qy_hat.argmax(dim=1, keepdim=True)
            te_accs += predicted.eq(Qy.view_as(predicted)).sum().item()
            
             # update metrics for epoch
            te_losses /= self.N * self.Q
            te_accs /= self.N * self.Q
        return te_losses, te_accs
    
    def test(self, dataloader):
        # iterative updates
        te_accs_total, te_losses_total = [], []
        for _ in trange(self.num_test_step, desc='[INFO] Meta-test', leave=False):
            # define new model for meta-test
            meta_tester = copy.deepcopy(self.model)

            # define model for meta training
            weights = OrderedDict((name, param) for (name, param) in meta_tester.named_parameters())

            # iterative updates
            te_losses, te_accs = 0.0, 0.0
            for _ in range(self.meta_test_step):
                for _ in range(self.B):           
                    # get dataset
                    Sx, Sy, Qx, Qy = next(iter(dataloader))
                    Sx, Sy = Sx.to(self.device).view(self.N * self.K, *Sx.shape[2:]), Sy.to(self.device).view(-1, 1).squeeze()
                    Qx, Qy = Qx.to(self.device).view(self.N * self.Q, *Qx.shape[2:]), Qy.to(self.device).view(-1, 1).squeeze()

                    # inference & get loss
                    Sy_hat = self.model(Sx)
                    tr_loss = self.criterion(Sy_hat, Sy)

                    # get support gradient & hessian
                    zero_grad(self.model.parameters())
                    support_grad = torch.autograd.grad(tr_loss, self.model.parameters(), create_graph=True)

                    # calculate updated weights (theta_prime)
                    adapted_weights = compute_adapted_weights(weights, support_grad, self.alpha)

                    # get loss on adapted parameter to new tasks (with computational graph attached)
                    Qy_hat = self.model(Qx, adapted_weights)
                    te_loss = self.criterion(Qy_hat, Qy)
                else:
                    # clear gradient cache
                    del support_grad
            else:
                # get metric and loss
                te_losses += te_loss.item()

                # get correct counts
                predicted = Qy_hat.argmax(dim=1, keepdim=True)
                te_accs += predicted.eq(Qy.view_as(predicted)).sum().item()

                 # update metrics for epoch
                te_losses /= self.N * self.Q
                te_accs /= self.N * self.Q

                # collect metrics for total steps
                te_losses_total.append(te_losses)
                te_accs_total.append(te_accs)
        else:
            # calculate mean losses
            te_losses_mean = np.asarray(te_losses_total).mean()
            
            # calculate mean accruacies
            te_accs_mean = np.asarray(te_accs_total).mean()
                
            # calculate CI constant for collected losses
            te_losses_ci = np.asarray(te_losses_total).std() * np.abs(scipy.stats.t.ppf((1. - 0.95) / 2, len(te_losses_total) - 1)) / np.sqrt(len(te_losses_total))
            
            # calculate CI constant for accuracies
            te_accs_ci = np.asarray(te_accs_total).std() * np.abs(scipy.stats.t.ppf((1. - 0.95) / 2, len(te_accs_total) - 1)) / np.sqrt(len(te_accs_total))    
        return te_losses_mean, te_losses_ci, te_accs_mean, te_accs_ci