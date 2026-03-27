import torch.nn as nn
import torch
from ... import build_from_configs
from .. import encoders
from ..decoders import VPOccDecoder
from ..losses import ce_ssc_loss, geo_scal_loss, sem_scal_loss
from .side_warping_symphonies import SideWarping
import os
from torchvision.utils import save_image
from ..warp_utils.warping_layers import CuboidGlobalKDEGrid, GaussianVPGrid, apply_unwarp
import torch.nn.functional as F

class VPOcc(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        image_shape=None,
        voxel_size=None,
        downsample_z=None,
        class_weights=None,
        criterions=None,
        skip_original=False,
        skip_warped=False,
        warp_type='side',
        use_unwarp=False,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.skip_original = skip_original
        self.skip_warped = skip_warped
        self.warp_type = warp_type
        self.use_unwarp = use_unwarp

        print("[vpocc.py] self.skip_original:", self.skip_original)
        print("[vpocc.py] self.skip_warped:", self.skip_warped)
        print("[vpocc.py] self.warp_type:", self.warp_type)
        print("[vpocc.py] self.use_unwarp:", self.use_unwarp)

        if self.skip_original and self.skip_warped:
            raise ValueError("Both skip_original and skip_warped cannot be True at the same time.")

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        self.decoder = VPOccDecoder(
            embed_dims,
            num_classes,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            downsample_z=downsample_z,
    )
        
        if self.warp_type == 'side':
            self.side_warping = SideWarping(image_shape=image_shape)
        elif self.warp_type == 'tpp':
            self.side_warping = CuboidGlobalKDEGrid(
                input_shape=tuple(image_shape),
                output_shape=tuple(image_shape),
                separable=True,
            )
            for p in self.side_warping.parameters():
                p.requires_grad = False
        elif self.warp_type == 'gaussian':
            self.side_warping = GaussianVPGrid(
                input_shape=tuple(image_shape),
                output_shape=tuple(image_shape),
                separable=True,
                sigma=180, # NOTE: hardcode for now, change later if needed
            )
        else:
            raise ValueError(f"Unsupported warp_type: {self.warp_type}")            
        

    def forward_original(self, inputs):
        origin_pred_insts = self.encoder(inputs['img'])
        origin_feats = origin_pred_insts.pop('feats')
        return origin_feats


    def forward_warped(self, inputs):

        # (3/26/26): ensure warping layers are on the same device as the input image (handles multi-GPU training where each GPU may have its own device)
        # Get the correct device for the current GPU
        device = inputs['img'].device

        # Force the hardcoded tensors in warping_layers to move to this device
        if hasattr(self.side_warping, 'filter'):
            self.side_warping.filter = self.side_warping.filter.to(device)
            if hasattr(self.side_warping, 'P_basis_x'):
                self.side_warping.P_basis_x = self.side_warping.P_basis_x.to(device)
                self.side_warping.P_basis_y = self.side_warping.P_basis_y.to(device)
            if hasattr(self.side_warping, 'P_basis'):
                self.side_warping.P_basis = self.side_warping.P_basis.to(device)


        # (3/26/26): switch between side warping and TPP (grid-based) warping
        if self.warp_type == 'side':
            warped_img, warp_dict = self.side_warping(inputs['img'], inputs)
        elif self.warp_type in ['tpp', 'gaussian']:
            grid = self.side_warping(inputs['img'], inputs['v_pts'])
            warped_img = F.grid_sample(inputs['img'],grid, align_corners=True)
            warp_dict = {'grid': grid}    


        warped_pred_insts = self.encoder(warped_img)

        # # # ===== Begin DEBUG =====
        # debug_dir = "debug"
        # os.makedirs(debug_dir, exist_ok=True)

        # print(f"[DEBUG] original img shape: {inputs['img'].shape}")
        # print(f"[DEBUG] warped img shape: {warped_img.shape}")

        # save_image(inputs['img'], os.path.join(debug_dir, "original.png"), normalize=True)
        # save_image(warped_img, os.path.join(debug_dir, "warped.png"), normalize=True)
        # # # ===== End DEBUG =====

        warped_feats = warped_pred_insts.pop('feats')

        # (3/27/26): unwarp features, and disable coord warping 
        if self.use_unwarp:
            warped_feats = [apply_unwarp(warp_dict['grid'], feat) for feat in warped_feats]
            
            # # # ===== Begin DEBUG =====
            # unwarped_img = apply_unwarp(warp_dict['grid'], warped_img)
            # save_image(unwarped_img, "debug/unwarped_img.png", normalize=True)
            # # # ===== End DEBUG =====

            warp_dict = None

        # # # ===== Begin DEBUG =====
        # print(f"[DEBUG] type(warped_feats): {type(warped_feats)}")
        # for i, feat in enumerate(warped_feats):
        #     fmap = feat[0].mean(dim=0, keepdim=True)  # [1, H, W]
        #     save_image(fmap, f"debug/feat_{i}.png", normalize=True)
        # exit()
        # # # ===== End DEBUG =====
        

        return warped_feats, warp_dict

    def forward(self, inputs):
        warp_dict = None
        warped_feats = None
        origin_feats = None


        # warped_img, warp_dict = self.side_warping(inputs['img'], inputs)
        
        # warped_pred_insts = self.encoder(warped_img)
        # warped_feats = warped_pred_insts.pop('feats')
        
        # origin_pred_insts = self.encoder(inputs['img'])
        # origin_feats = origin_pred_insts.pop('feats')
        
        # (3/25/26): compute features per branch only if available
        if not self.skip_warped:
            warped_feats, warp_dict = self.forward_warped(inputs)

        if not self.skip_original:
            origin_feats = self.forward_original(inputs)


        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))

        feats_dict = {'origin_feats': origin_feats, 'warped_feats': warped_feats}
        vanishing_point = inputs['v_pts'][0,:]
        
        output = self.decoder(vanishing_point, feats_dict, depth, K, E, voxel_origin, projected_pix, fov_mask, warp_dict)
        return {'ssc_logits': output}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        
        for loss in self.criterions:
            losses['loss_' + loss] = loss_map[loss](preds, target)

        return losses