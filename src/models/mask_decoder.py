import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union
from .unet import Unet_decoder, Conv, TwoConv
from GeodisTK import geodesic3d_raster_scan as Geo3d
from monai.networks.nets import UNet
import numpy as np


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        args,
        *,
        transformer_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        multiple_outputs: bool = False,
        num_multiple_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.args = args
        self.multiple_outputs = multiple_outputs
        self.num_multiple_outputs = num_multiple_outputs
        # if self.args.use_sam3d_turbo:
        #     self.output_hypernetworks_mlps = nn.ModuleList([MLP(384, 384, 48, 3) for i in range(num_multiple_outputs + 1)])
        # else:
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(384, 384, 32, 3) for i in range(num_multiple_outputs + 1)])
        self.iou_prediction_head = MLP(384, 256, num_multiple_outputs + 1, 3, sigmoid_output=True)

        #self.final_conv = Conv["conv", 3](in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0)


        # FIXME below is the code runing "cats_multiple_outputs_2loss"
        # if multiple_outputs:
        #     self.output_hypernetworks_mlps = nn.ModuleList([MLP(384, 384, 32, 3) for i in range(3)])
        #     self.iou_prediction_head = MLP(384, 256, 3, 3, sigmoid_output=True)
        #     self.final_conv = Conv["conv", 3](in_channels=3, out_channels=3, kernel_size=1)
        # else:
        #     self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, 32, 3)
        #     self.final_conv = Conv["conv", 3](in_channels=1, out_channels=1, kernel_size=1)

        # self.expand_block = nn.Sequential(
        #     Conv["conv", 3](in_channels=32, out_channels=128, kernel_size=1, groups=4),
        #     # nn.InstanceNorm3d(128), nn.LeakyReLU(True)
        # )


        self.decoder = Unet_decoder(spatial_dims=3, features=(32, 32, 64, 128, 384, 32))

        if self.args.refine:
            self.refine = Refine(self.args)
        # self.refine = Refine_unet()
        # self.refine_conv = Conv["conv", 3](in_channels=32, out_channels=1, kernel_size=1)

    def forward(
        self,
        prompt_embeddings: torch.Tensor, # prompt_embedding --> [b, self.num_mask_tokens, c]
        image_embeddings, # image_embedding --> [b, c, low_res / 4, low_res / 4, low_res / 4]
        feature_list: List[torch.Tensor],
        image_level_features: torch.Tensor,
        image: torch.Tensor,
        points: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        upscaled_embedding = self.decoder(image_embeddings, feature_list)
        #upscaled_embedding = self.expand_block(upscaled_embedding)
        masks, iou_pred = self._predict_mask(upscaled_embedding, prompt_embeddings)
        return masks, iou_pred


    def _predict_mask(self, upscaled_embedding, prompt_embeddings):
        b, c, x, y, z = upscaled_embedding.shape

        iou_token_out = prompt_embeddings[:, 0, :]
        mask_tokens_out = prompt_embeddings[:, 1: (self.num_multiple_outputs + 1 + 1), :]  # multiple masks + iou

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_multiple_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        #masks = self.final_conv(masks) + masks
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.multiple_outputs:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

        # FIXME below is the code runing "cats_multiple_outputs_2loss"
        # if self.multiple_outputs:
        #     iou_token_out = prompt_embeddings[:, 0, :]
        #     mask_tokens_out = prompt_embeddings[:, 1: (1 + 3), :]
        #     hyper_in_list: List[torch.Tensor] = []
        #     for i in range(3):
        #         hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        #     hyper_in = torch.stack(hyper_in_list, dim=1)
        #     iou_pred = self.iou_prediction_head(iou_token_out)
        # else:
        #     iou_pred = None
        #     mask_tokens_out = prompt_embeddings[:, 0, :].unsqueeze(1)
        #     hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
        # mask = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        # mask = self.final_conv(mask) + mask
        # return mask, iou_pred



        # """
        # original
        # b, c, x, y, z = upscaled_embedding.shape
        # factor = self.output_hypernetworks_mlps(prompt_embeddings)
        # mask = (factor @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        # mask = self.final_conv(mask) + mask


        # refine_input = self._get_refine_input(image, mask, points)
        # mask_refine = self.refine(refine_input)
        #return mask, mask
        # """

        # plain SAM
        # upscaled_embedding = self.output_upscaling(image_embeddings)
        #
        # b, c, x, y, z = upscaled_embedding.shape
        # factor = self.output_hypernetworks_mlps(prompt_embeddings)
        # mask = (factor @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        # mask = self.final_conv(mask)
        # return mask

class Refine_unet(nn.Module):
    def __init__(self):
        super(Refine_unet, self).__init__()
        self.refine = UNet(spatial_dims=3, in_channels=4, out_channels=1,
                           channels=(32, 64, 64), strides=(2, 2), num_res_units=2)
    def forward(self, x):
        return self.refine(x)

class Refine(nn.Module):
    def __init__(self,
                 args,
                 spatial_dims: int = 3,
                 in_channel: int = 4,
                 out_channel: int = 32,
                 act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: Union[str, tuple] = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: Union[float, tuple] = 0.0,
                 ):
        super().__init__()
        self.args = args

        self.first_conv = Conv["conv", 3](in_channels=in_channel, out_channels=out_channel, kernel_size=1)

        self.conv1 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)
        self.conv2 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)
        # self.conv3 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)

        self.conv_error_map = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)
        self.conv_correction = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)


    def forward(self, image, mask_best, points, mask):
        # x, crop_size = self._get_refine_input(image, mask_best, points)
        # c_x, c_y, c_z = crop_size[0], crop_size[1], crop_size[2]
        # if self.args.refine_crop:
        #     if max(c_x, c_y, c_z) == 128:
        #         mask = F.interpolate(mask, scale_factor=0.5, mode='trilinear', align_corners=False)
        #     else:
        #         mask = mask[:, :, int(64 - c_x / 2): int(64 + c_x / 2), int(64 - c_y / 2): int(64 + c_y / 2), int(64 - c_z / 2): int(64 + c_z / 2)]
        # else:
        #     mask = F.interpolate(mask, scale_factor=0.5, mode='trilinear', align_corners=False)
        #
        # print(mask.shape, crop_size)

        x = self._get_refine_input(image, mask_best, points)
        mask = F.interpolate(mask, scale_factor=0.5, mode='trilinear', align_corners=False)

        x = self.first_conv(x)

        residual = x
        x = self.conv1(x)
        x = residual + x

        residual = x
        x = self.conv2(x)
        x = residual + x

        # residual = x
        # x = self.conv3(x)
        # x = residual + x
        error_map = self.conv_error_map(x)
        correction = self.conv_correction(x)

        outputs = (error_map * correction + mask)

        # if self.args.refine_crop:
        #     if max(crop_size) == 128:
        #         outputs_return = F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False)
        #     else:
        #         outputs_return = torch.zeros_like(image) # FIXME should change it to mask
        #         outputs_return[:, :, int(64 - c_x/2): int(64 + c_x/2),
        #                              int(64 - c_y/2): int(64 + c_y/2),
        #                              int(64 - c_z/2): int(64 + c_z/2)] = outputs # FIXME should change it to mask
        # else:
        #     outputs_return = F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False)

        outputs = F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False)
        error_map = F.interpolate(error_map, scale_factor=2, mode='trilinear', align_corners=False)
        return outputs, error_map

    def _get_refine_input(self, image, mask, points):
        mask = torch.sigmoid(mask)
        mask = (mask > 0.5)

        coors, labels = points[0], points[1]
        positive_map, negative_map = torch.zeros_like(image), torch.zeros_like(image)

        for click_iters in range(len(coors)):
            coors_click, labels_click = coors[click_iters], labels[click_iters]
            for batch in range(image.size(0)):
                point_label = labels_click[batch]
                coor = coors_click[batch]
                # sepehre_coor = [coor[:, 0], coor[:, 1], coor[:, 2]]

                # Create boolean masks
                negative_mask = point_label == 0
                positive_mask = point_label != 0

                # Update negative_map
                if negative_mask.any():  # Check if there's at least one True in negative_mask
                    negative_indices = coor[negative_mask]
                    for idx in negative_indices:
                        negative_map[batch, 0, idx[0], idx[1], idx[2]] = 1

                # Update positive_map
                if positive_mask.any():  # Check if there's at least one True in negative_mask
                    positive_indices = coor[positive_mask]
                    for idx in positive_indices:
                        positive_map[batch, 0, idx[0], idx[1], idx[2]] = 1


                # if point_label == 0:
                #     negative_map[batch, :, coor[:, 0], coor[:, 1], coor[:, 2]] = 1
                #     #negative_map[batch, 0, :] = self._get_sephere(sepehre_coor, negative_map[batch, 0, :])
                # else:
                #     positive_map[batch, :, coor[:, 0], coor[:, 1], coor[:, 2]] = 1
                #     #positive_map[batch, 0, :] = self._get_sephere(sepehre_coor, positive_map[batch, 0, :])

        # all_map = (positive_map + negative_map).astype(torch.uint8)
        # geo = torch.zeros_like(image)
        # for batch in range(image.size(0)):
        #     geo_output = torch.tensor(Geo3d(image[batch, :].permute(3, 2, 1, 0).cpu().numpy(),
        #                                     all_map[batch, :].permute(3, 2, 1, 0).cpu().numpy(), [1, 1, 1], 1, 4))
        #     geo[batch, 0, :] = geo_output.permute(2, 1, 0)
        #refine_input = torch.cat([image, mask, positive_map, negative_map, geo.to(image.device)], dim=1)


        # refine_input = torch.cat([image, mask, positive_map, negative_map], dim=1)

        refine_input = F.interpolate(torch.cat([image, mask, positive_map, negative_map], dim=1), scale_factor=0.5, mode='trilinear')
        return refine_input

        # refine_input_crop, crop_size = self._crop(refine_input, negative_map, positive_map)
        # if self.args.refine_crop:
        #     if refine_input_crop.size == refine_input.size:
        #         mask_info = F.interpolate(torch.cat([mask, positive_map, negative_map], dim=1), scale_factor=0.5, mode='nearest')
        #         refine_input = torch.cat([F.interpolate(image, scale_factor=0.5, mode='trilinear', align_corners=False), mask_info], dim=1)
        #         return refine_input, crop_size
        #     else:
        #         return refine_input_crop, crop_size
        # else:
        #     mask_info = F.interpolate(torch.cat([mask, positive_map, negative_map], dim=1), scale_factor=0.5, mode='nearest')
        #     refine_input = torch.cat(
        #         [F.interpolate(image, scale_factor=0.5, mode='trilinear', align_corners=False), mask_info], dim=1)
        #     return refine_input, crop_size

    def _crop(self, refine_input, negative_map, positive_map):
        map = negative_map + positive_map
        from src.utils.util import _bbox_mask

        bbox_coords = _bbox_mask(map[:, 0, :])
        b = bbox_coords[:, :, 3:6] - bbox_coords[:, :, 0:3]

        max_dimension, _ = torch.max(b, dim=0)

        c_x = self._get_max_dimension(max_dimension[:,0])
        c_y = self._get_max_dimension(max_dimension[:,1])
        c_z = self._get_max_dimension(max_dimension[:,2])


        if max(c_x, c_y, c_z)  == 128:
            return refine_input, (c_x, c_y, c_z)
        else:
            refine_input_crop = refine_input[:, :, int(64 - c_x/2): int(64 + c_x/2),
                                                   int(64 - c_y/2): int(64 + c_y/2),
                                                   int(64 - c_z/2): int(64 + c_z/2)]
            return refine_input_crop, (c_x, c_y, c_z)

    def _get_max_dimension(self, max_dimension):
        if max_dimension <= 32:
            crop_size = 32
        elif max_dimension > 32 and max_dimension <= 64:
            crop_size = 64
        elif max_dimension > 64 and max_dimension < 96:
            crop_size = 96
        else:
            crop_size = 128
        return crop_size


    def _get_sephere(self, points, maps, radius=3):
        coords = torch.meshgrid(
            torch.arange(maps.shape[0]), torch.arange(maps.shape[1]), torch.arange(maps.shape[2]))
        x, y, z = coords
        x, y, z = x.to(maps.device), y.to(maps.device), z.to(maps.device)
        distance_from_center = torch.sqrt((x - points[0]) ** 2 + (y - points[1]) ** 2 + (z - points[2]) ** 2)
        diameter = (distance_from_center <= radius).int()
        return diameter


class Upsampling(nn.Module):
    def __init__(self, in_channels=384, out_channels=128):
        super(Upsampling, self).__init__()
        self.up = nn.Sequential(
                     nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(out_channels),
                     nn.ReLU(),
                     # nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2),
                     # nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                     # nn.InstanceNorm3d(out_channels),
                     # nn.ReLU(),
                     nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2),
                     nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(out_channels),
                     nn.ReLU(),
                     nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2),
                     nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(out_channels),
                     nn.ReLU(),
                                   )
    def forward(self, x):
        return self.up(x)
        #return torch.cat([head2, head3, head4, head5], dim=1)


class Decoder(nn.Module):
    def __init__(self, in_channels=256, mlahead_channels=128, num_classes=1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.mla_channels = in_channels
        self.mlahead_channels = mlahead_channels

        self.up0 = Upsampling()
        self.up1 = Upsampling()
        self.up2 = Upsampling()
        self.up3 = Upsampling()

        self.head = nn.Sequential(
                     nn.Conv3d(2 * mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     #nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False)
                     nn.Conv3d(mlahead_channels, 48, 3, padding=1, bias=False)
        )

        self.CNN_encoder = nn.Sequential(
                     nn.Conv3d(2, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     )

    def _multi_feature_upsampling(self, features):
        # f0 = self.up0(features[0])
        # f1 = self.up1(features[1])
        # f2 = self.up2(features[2])
        # f3 = self.up3(features[3])
        # return torch.cat([f0, f1, f2, f3], dim=1)

        return self.up3(features[3])

    def forward(self, features, image_level_feature):
        x = self._multi_feature_upsampling(features)
        image_features = self.CNN_encoder(image_level_feature)
        # x = torch.cat([x, image_features, image_level_feature], dim=1)
        x = torch.cat([x, image_features], dim=1)

        x = self.head(x)
        #x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

