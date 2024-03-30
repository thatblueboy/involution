import torch.nn as nn
from utils.involution import Involution
class Bottleneck(nn.Module):
  """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

  def __init__(self, in_channels, out_channels, expansion = 4, downsample=None, stride=1, has_involution=False):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.expansion = expansion
        mid_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)#, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = Involution(mid_channels, kernel_size=3, stride=1) if has_involution else nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)#, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
  def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion

class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down = False,
                 is_rednet = False,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=conv_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block( in_channels=in_channels, out_channels=out_channels,expansion=self.expansion, stride=stride, downsample=downsample, has_involution = is_rednet))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    ))
        # self.add_module('layer', nn.Sequential(*layers))

        super(ResLayer, self).__init__(*layers)

class ResNet(nn.Module):
  arch_settings = {
    26: (Bottleneck, (1, 2, 4, 1)),
    38: (Bottleneck, (2, 3, 5, 2)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))
}


  def __init__(self, 
                depth,
              in_channels=3,
              stem_channels=64,
              base_channels=64,
              expansion = None,
              num_stages=4,
              strides=(1, 2, 2, 2),
              out_indices=(3, ),
              frozen_stages=-1,
              avg_down=False,
              zero_init_residual=True,
              is_rednet = False,
              ):
    super(ResNet, self).__init__()
    if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
    self.avg_down = avg_down
    self.depth = depth
    self.stem_channels = stem_channels
    self.base_channels = base_channels
    self.frozen_stages = frozen_stages

    self.num_stages = num_stages
    assert num_stages >= 1 and num_stages <= 4
    self.strides = strides
    self.out_indices = out_indices
    assert max(out_indices) < num_stages
    self.zero_init_residual = zero_init_residual
    self.block, stage_blocks = self.arch_settings[depth]
    self.stage_blocks = stage_blocks[:num_stages]
    self.expansion = get_expansion(self.block, expansion)

    self._make_stem_layer(in_channels, stem_channels)

    self.res_layers = []

    _in_channels = stem_channels
    _out_channels = base_channels * self.expansion

    for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                avg_down=self.avg_down,
                is_rednet = is_rednet
                )
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
    self._freeze_stages()         
    self.feat_dim = res_layer[-1].out_channels


  def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)
  @property
  def norm1(self):
        return getattr(self, self.norm1_name)

  def _freeze_stages(self):
      if self.frozen_stages >= 0:
          self.stem.eval()
          for param in self.stem.parameters():
              param.requires_grad = False
      for i in range(1, self.frozen_stages + 1):
          m = getattr(self, f'layer{i}')
          m.eval()
          for param in m.parameters():
              param.requires_grad = False

  def init_weights(self, pretrained=None):
    super(ResNet, self).init_weights(pretrained)

    if pretrained is None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)
                    nn.init.constant_(m.norm3.bias, 0)


  def _make_stem_layer(self, in_channels, stem_channels):
    self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels//2, stem_channels, kernel_size=3, stride=1, padding=1),
        )

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

  def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
