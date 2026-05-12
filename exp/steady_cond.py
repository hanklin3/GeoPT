import os
import csv
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss
from utils.visual import visual, visual_reynolds_cavitation_2d_preview, visual_reynolds_cavitation_3d_preview
import matplotlib.pyplot as plt
import numpy as np
import math


class Exp_Steady(Exp_Basic):

    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)

    def _append_train_log(self, epoch, train_loss, val_loss):
        if not self.is_main_process:
            return
        os.makedirs('./training_logs', exist_ok=True)
        log_path = os.path.join('./training_logs', self.args.save_name + '_metrics.csv')
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch, train_loss, val_loss])

    def _write_eval_log(self, tag, metrics):
        if not self.is_main_process:
            return
        os.makedirs('./training_logs', exist_ok=True)
        log_path = os.path.join('./training_logs', self.args.save_name + '_eval.log')
        with open(log_path, 'a') as handle:
            handle.write(f'[{tag}]\n')
            for key, value in metrics.items():
                handle.write(f'{key}: {value}\n')
            handle.write('\n')

    def _pointwise_relative_error(self, out, y, eps=1e-8):
        denom = torch.clamp(torch.abs(y), min=eps)
        return torch.abs(out - y) / denom

    def _channel_relative_l2(self, out, y, eps=1e-8):
        error_norm = torch.sqrt(torch.sum((out - y) ** 2, dim=(0, 1)))
        target_norm = torch.sqrt(torch.sum(y ** 2, dim=(0, 1))).clamp_min(eps)
        return error_norm / target_norm

    def _reynolds2d_constraint_metrics(self, pos_np, field_np, cond_np):
        xs = np.unique(pos_np[:, 0])
        zs = np.unique(pos_np[:, 1])
        nx, nz = len(xs), len(zs)
        if nx * nz != pos_np.shape[0] or nx < 3 or nz < 3:
            return None

        p = field_np[:, 0].reshape(nx, nz)
        rho = field_np[:, 2].reshape(nx, nz)
        h = pos_np[:, 2].reshape(nx, nz)
        speed = float(cond_np[0])
        viscosity = float(cond_np[1])

        dx = float(np.mean(np.diff(xs)))
        dz = float(np.mean(np.diff(zs)))
        coeff = rho * h ** 3 / (12.0 * viscosity + 1e-12)
        dpdx = np.gradient(p, dx, axis=0, edge_order=2)
        dpdz = np.gradient(p, dz, axis=1, edge_order=2)
        flux_x = coeff * dpdx
        flux_z = coeff * dpdz
        div_flux = (
            np.gradient(flux_x, dx, axis=0, edge_order=2)
            + np.gradient(flux_z, dz, axis=1, edge_order=2)
        )
        source = 0.5 * speed * np.gradient(rho * h, dx, axis=0, edge_order=2)
        residual = div_flux - source

        interior = residual[1:-1, 1:-1]
        scale = np.mean(np.abs(div_flux[1:-1, 1:-1])) + np.mean(np.abs(source[1:-1, 1:-1])) + 1e-12
        return {
            'reynolds_residual_l1': float(np.mean(np.abs(interior))),
            'reynolds_residual_l2': float(np.sqrt(np.mean(interior ** 2))),
            'reynolds_residual_linf': float(np.max(np.abs(interior))),
            'reynolds_residual_relative_l1': float(np.mean(np.abs(interior)) / scale),
        }

    def _write_reynolds2d_metrics_report(self, output_dir, csv_path, rows):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'per_sample_metrics.md')
        rel_l2 = np.array([row['rel_l2'] for row in rows], dtype=np.float64)
        pred_res = np.array([row['pred_reynolds_residual_relative_l1'] for row in rows], dtype=np.float64)
        true_res = np.array([row['true_reynolds_residual_relative_l1'] for row in rows], dtype=np.float64)

        with open(report_path, 'w') as handle:
            handle.write('# Reynolds2D Per-Sample Metrics\n\n')
            handle.write(f'- CSV: `{os.path.basename(csv_path)}`\n')
            handle.write(f'- Samples: {len(rows)}\n')
            handle.write(f'- Mean relative L2 prediction error: {rel_l2.mean():.6g}\n')
            handle.write(f'- Mean predicted relative conservation residual L1: {pred_res.mean():.6g}\n')
            handle.write(f'- Mean ground-truth relative conservation residual L1: {true_res.mean():.6g}\n\n')

            handle.write('## Conservation Form\n\n')
            handle.write('The residual checks the density-aware Reynolds equation used by the synthetic generator:\n\n')
            handle.write('```text\n')
            handle.write('R = div((rho h^3 / (12 mu)) grad p) - (V / 2) d(rho h) / dx\n')
            handle.write('```\n\n')
            handle.write('where:\n\n')
            handle.write('- `p` is pressure\n')
            handle.write('- `rho` is mixture density\n')
            handle.write('- `h` is film height from the input test geometry\n')
            handle.write('- `mu` is viscosity from `cond_*.npy`\n')
            handle.write('- `V` is wall speed from `cond_*.npy`\n\n')

            handle.write('## Residual Metrics\n\n')
            handle.write('For each full-mesh test sample, the report computes the residual on the interior grid points:\n\n')
            handle.write('```text\n')
            handle.write('residual_l1 = mean(abs(R))\n')
            handle.write('residual_l2 = sqrt(mean(R^2))\n')
            handle.write('residual_linf = max(abs(R))\n')
            handle.write('residual_relative_l1 = mean(abs(R)) / (mean(abs(div_flux)) + mean(abs(source)) + eps)\n')
            handle.write('```\n\n')
            handle.write('Both predicted and ground-truth residuals are included in the CSV as `pred_*` and `true_*` columns.\n')

            handle.write('\n## Per-Sample Table\n\n')
            handle.write('| Sample | Points | Rel L2 | MSE | MAE | Pred Rel Residual L1 | True Rel Residual L1 |\n')
            handle.write('|---:|---:|---:|---:|---:|---:|---:|\n')
            for row in rows:
                handle.write(
                    f"| {row['sample_id']} "
                    f"| {row['points']} "
                    f"| {row['rel_l2']:.6g} "
                    f"| {row['mse']:.6g} "
                    f"| {row['mae']:.6g} "
                    f"| {row['pred_reynolds_residual_relative_l1']:.6g} "
                    f"| {row['true_reynolds_residual_relative_l1']:.6g} |\n"
                )

            handle.write('\n## Channel Error Table\n\n')
            handle.write('| Sample | Pressure Rel L2 | Vapor Rel L2 | Density Rel L2 | Film Height Rel L2 | Shear Rel L2 |\n')
            handle.write('|---:|---:|---:|---:|---:|---:|\n')
            for row in rows:
                handle.write(
                    f"| {row['sample_id']} "
                    f"| {row['pressure_rel_l2']:.6g} "
                    f"| {row['vapor_fraction_rel_l2']:.6g} "
                    f"| {row['density_rel_l2']:.6g} "
                    f"| {row['film_height_rel_l2']:.6g} "
                    f"| {row['shear_proxy_rel_l2']:.6g} |\n"
                )

        print(f"Saved Reynolds2D metrics report to: {os.path.abspath(report_path)}")

    def _write_reynolds3d_metrics_report(self, output_dir, csv_path, rows):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'per_sample_metrics.md')
        rel_l2 = np.array([row['rel_l2'] for row in rows], dtype=np.float64)
        channel_names = ['u_velocity', 'v_velocity', 'pressure', 'vapor_fraction', 'density']

        with open(report_path, 'w') as handle:
            handle.write('# Reynolds3D Per-Sample Metrics\n\n')
            handle.write(f'- CSV: `{os.path.basename(csv_path)}`\n')
            handle.write(f'- Samples: {len(rows)}\n')
            handle.write(f'- Mean relative L2 prediction error: {rel_l2.mean():.6g}\n\n')

            handle.write('## Target Channels\n\n')
            handle.write('The 3D Reynolds target vector is `[u, v, p, alpha_v, rho]`:\n\n')
            handle.write('- `u`: sliding-direction velocity\n')
            handle.write('- `v`: transverse in-plane velocity\n')
            handle.write('- `p`: pressure after cavitation clipping\n')
            handle.write('- `alpha_v`: vapor fraction\n')
            handle.write('- `rho`: liquid-vapor mixture density\n\n')

            handle.write('## Per-Sample Table\n\n')
            handle.write('| Sample | Points | Rel L2 | MSE | MAE |\n')
            handle.write('|---:|---:|---:|---:|---:|\n')
            for row in rows:
                handle.write(
                    f"| {row['sample_id']} "
                    f"| {row['points']} "
                    f"| {row['rel_l2']:.6g} "
                    f"| {row['mse']:.6g} "
                    f"| {row['mae']:.6g} |\n"
                )

            handle.write('\n## Channel Error Table\n\n')
            handle.write('| Sample | U Rel L2 | V Rel L2 | Pressure Rel L2 | Vapor Rel L2 | Density Rel L2 |\n')
            handle.write('|---:|---:|---:|---:|---:|---:|\n')
            for row in rows:
                handle.write(
                    f"| {row['sample_id']} "
                    f"| {row['u_velocity_rel_l2']:.6g} "
                    f"| {row['v_velocity_rel_l2']:.6g} "
                    f"| {row['pressure_rel_l2']:.6g} "
                    f"| {row['vapor_fraction_rel_l2']:.6g} "
                    f"| {row['density_rel_l2']:.6g} |\n"
                )

            handle.write('\n## Mean Channel Errors\n\n')
            for name in channel_names:
                values = np.array([row[f'{name}_rel_l2'] for row in rows], dtype=np.float64)
                handle.write(f'- `{name}` mean relative L2: {values.mean():.6g}\n')

        print(f"Saved Reynolds3D metrics report to: {os.path.abspath(report_path)}")

    def _append_per_sample_metrics(self, tag, rows):
        if not self.is_main_process or not rows:
            return
        if self.args.loader in ('ReynoldsCavitation2D', 'ReynoldsCavitation3D') and tag == 'test_full_mesh':
            output_dir = os.path.join('./results', self.args.save_name, 'test_full_previews', 'metrics')
        else:
            output_dir = './training_logs'
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f'{self.args.save_name}_{tag}_per_sample.csv')
        fieldnames = list(rows[0].keys())
        with open(log_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-sample metrics to: {os.path.abspath(log_path)}")
        if self.args.loader == 'ReynoldsCavitation2D' and tag == 'test_full_mesh':
            self._write_reynolds2d_metrics_report(output_dir, log_path, rows)
        if self.args.loader == 'ReynoldsCavitation3D' and tag == 'test_full_mesh':
            self._write_reynolds3d_metrics_report(output_dir, log_path, rows)

    def vali(self):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                x = pos.to(self.device)
                fx = fx.to(self.device)
                cond = cond.to(self.device)
                y = y.to(self.device)
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :self.args.space_dim], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= self.args.ntest
        return rel_err

    def train(self):
        ### load GeoPT pre-trained model
        if self.args.finetune:
            self.model = self.load_pretrained_with_filter(self.model,
                                                          "./checkpoints/" + self.args.finetune_name + ".pt")
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Optimizer only AdamW or Adam')

        ### adopt learning rate scheduler
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        myloss = L2Loss(size_average=False)

        train_loss_list = []
        test_loss_list = []

        for ep in range(self.args.epochs):
            self.set_epoch(ep)

            self.model.train()
            train_loss = 0

            for pos, fx, cond, y in self.train_loader:
                x = pos.to(self.device)
                fx = fx.to(self.device)
                cond = cond.to(self.device)
                y = y.to(self.device)
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :self.args.space_dim], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                    y = self.dataset.y_normalizer.decode(y)
                loss = myloss(out, y)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            if self.args.memory_report_interval > 0 and (ep % self.args.memory_report_interval == 0):
                self.report_gpu_memory(tag=f'epoch_{ep}')

            train_loss = train_loss / self.args.ntrain
            if self.is_main_process:
                print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))
            train_loss_list.append(train_loss)

            rel_err = self.vali()
            if self.is_main_process:
                print("rel_err:{}".format(rel_err))
            test_loss_list.append(rel_err)
            self._append_train_log(ep, train_loss, rel_err)

            if ep % 100 == 0 and self.is_main_process:
                print('save models')
            if ep % 100 == 0:
                self.save_checkpoint(os.path.join('./checkpoints', self.args.save_name + '.pt'))

            if ep % 10 == 0 and self.is_main_process:
                if not os.path.exists('./training_logs'):
                    os.makedirs('./training_logs')
                print('save logs')
                np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'),
                        np.array(train_loss_list))
                np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'),
                        np.array(test_loss_list))

        if self.is_main_process:
            print('final save models')
        self.save_checkpoint(os.path.join('./checkpoints', self.args.save_name + '.pt'))
        if self.is_main_process:
            if not os.path.exists('./training_logs'):
                os.makedirs('./training_logs')
            print('final training logs')
            np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'), np.array(train_loss_list))
            np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'), np.array(test_loss_list))
            self._write_eval_log('final_train', {
                'final_train_loss': train_loss_list[-1] if train_loss_list else None,
                'final_val_loss': test_loss_list[-1] if test_loss_list else None,
            })

    def test(self):
        self.load_checkpoint("./checkpoints/" + self.args.save_name + ".pt")
        if not self.is_main_process:
            return
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                id += 1
                x = pos.to(self.device)
                fx = fx.to(self.device)
                cond = cond.to(self.device)
                y = y.to(self.device)
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :self.args.space_dim], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                point_rel = self._pointwise_relative_error(out, y)
                rel_err_split += self._channel_relative_l2(out, y)
                rel_err_split_max += torch.max(torch.max(point_rel, dim=0)[0], dim=0)[0]
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)
                    if self.args.loader == 'ReynoldsCavitation2D':
                        visual_reynolds_cavitation_2d_preview(x, fx, y, self.args, id, split='test')
                    if self.args.loader == 'ReynoldsCavitation3D':
                        visual_reynolds_cavitation_3d_preview(x, y, out, self.args, id, split='test', cond=cond)

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))
        self._write_eval_log('test', {
            'test_rel_err': rel_err,
            'test_mse': mse,
            'test_mae': mae,
            'test_rel_err_split': rel_err_split.detach().cpu().tolist() if torch.is_tensor(rel_err_split) else rel_err_split,
            'test_rel_err_split_max': rel_err_split_max.detach().cpu().tolist() if torch.is_tensor(rel_err_split_max) else rel_err_split_max,
        })

    def test_full_mesh(self):
        self.load_checkpoint("./checkpoints/" + self.args.save_name + ".pt")
        if not self.is_main_process:
            return
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)
        per_sample_rows = []

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader_full:
                id += 1
                x = pos.to(self.device)
                fx = fx.to(self.device)
                cond = cond.to(self.device)
                y = y.to(self.device)
                fx = torch.cat((fx, cond.repeat(1, fx.shape[1], 1)), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :self.args.space_dim], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                point_rel = self._pointwise_relative_error(out, y)
                rel_err_split += self._channel_relative_l2(out, y)
                rel_err_split_max += torch.max(torch.max(point_rel, dim=0)[0], dim=0)[0]
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)
                    if self.args.loader == 'ReynoldsCavitation2D':
                        visual_reynolds_cavitation_2d_preview(x, fx, y, self.args, id, split='test_full')
                    if self.args.loader == 'ReynoldsCavitation3D':
                        visual_reynolds_cavitation_3d_preview(x, y, out, self.args, id, split='test_full', cond=cond)

                if self.args.loader == 'ReynoldsCavitation2D':
                    out_np = out.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()
                    pos_np = x.detach().cpu().numpy()
                    cond_np = cond.detach().cpu().numpy()
                    batch_size = out_np.shape[0]
                    for batch_idx in range(batch_size):
                        sample_id = id + batch_idx
                        err = out_np[batch_idx] - y_np[batch_idx]
                        channel_mse = np.mean(err ** 2, axis=0)
                        channel_mae = np.mean(np.abs(err), axis=0)
                        channel_rel_l2 = np.sqrt(np.sum(err ** 2, axis=0)) / (
                            np.sqrt(np.sum(y_np[batch_idx] ** 2, axis=0)) + 1e-12
                        )
                        row = {
                            'sample_id': sample_id,
                            'points': int(out_np.shape[1]),
                            'rel_l2': float(np.sqrt(np.sum(err ** 2)) / (np.sqrt(np.sum(y_np[batch_idx] ** 2)) + 1e-12)),
                            'mse': float(np.mean(err ** 2)),
                            'mae': float(np.mean(np.abs(err))),
                        }
                        names = ['pressure', 'vapor_fraction', 'density', 'film_height', 'shear_proxy']
                        for channel_idx, name in enumerate(names):
                            row[f'{name}_mse'] = float(channel_mse[channel_idx])
                            row[f'{name}_mae'] = float(channel_mae[channel_idx])
                            row[f'{name}_rel_l2'] = float(channel_rel_l2[channel_idx])

                        pred_constraint = self._reynolds2d_constraint_metrics(
                            pos_np[batch_idx], out_np[batch_idx], cond_np[batch_idx, 0]
                        )
                        true_constraint = self._reynolds2d_constraint_metrics(
                            pos_np[batch_idx], y_np[batch_idx], cond_np[batch_idx, 0]
                        )
                        if pred_constraint is not None:
                            for key, value in pred_constraint.items():
                                row[f'pred_{key}'] = value
                        if true_constraint is not None:
                            for key, value in true_constraint.items():
                                row[f'true_{key}'] = value
                        per_sample_rows.append(row)
                    id += batch_size - 1
                elif self.args.loader == 'ReynoldsCavitation3D':
                    out_np = out.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()
                    batch_size = out_np.shape[0]
                    for batch_idx in range(batch_size):
                        sample_id = id + batch_idx
                        err = out_np[batch_idx] - y_np[batch_idx]
                        channel_mse = np.mean(err ** 2, axis=0)
                        channel_mae = np.mean(np.abs(err), axis=0)
                        channel_rel_l2 = np.sqrt(np.sum(err ** 2, axis=0)) / (
                            np.sqrt(np.sum(y_np[batch_idx] ** 2, axis=0)) + 1e-12
                        )
                        row = {
                            'sample_id': sample_id,
                            'points': int(out_np.shape[1]),
                            'rel_l2': float(np.sqrt(np.sum(err ** 2)) / (np.sqrt(np.sum(y_np[batch_idx] ** 2)) + 1e-12)),
                            'mse': float(np.mean(err ** 2)),
                            'mae': float(np.mean(np.abs(err))),
                        }
                        names = ['u_velocity', 'v_velocity', 'pressure', 'vapor_fraction', 'density']
                        for channel_idx, name in enumerate(names):
                            row[f'{name}_mse'] = float(channel_mse[channel_idx])
                            row[f'{name}_mae'] = float(channel_mae[channel_idx])
                            row[f'{name}_rel_l2'] = float(channel_rel_l2[channel_idx])
                        per_sample_rows.append(row)
                    id += batch_size - 1

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))
        self._write_eval_log('test_full_mesh', {
            'test_rel_err': rel_err,
            'test_mse': mse,
            'test_mae': mae,
            'test_rel_err_split': rel_err_split.detach().cpu().tolist() if torch.is_tensor(rel_err_split) else rel_err_split,
            'test_rel_err_split_max': rel_err_split_max.detach().cpu().tolist() if torch.is_tensor(rel_err_split_max) else rel_err_split_max,
        })
        self._append_per_sample_metrics('test_full_mesh', per_sample_rows)
