import torch
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml
from torch.nn.init import normal, constant, xavier_uniform


class ECO(nn.Module):
    def __init__(self, model_path='tf_model_zoo/ECO/ECO.yaml', num_classes=101,
                       weight_url_2d='http://pa0630vji.bkt.gdipper.com/zhangcan/pth/models/bninception_rgb_kinetics_init-d4ee618d3399.pth',
                       weight_url_3d='http://pa0630vji.bkt.gdipper.com/zhangcan/pth/models/resnet-18-kinetics_init-ead73157.pth',
                       num_segments=4, pretrained_parts='both'):
        super(ECO, self).__init__()

        self.num_segments = num_segments

        self.pretrained_parts = pretrained_parts

        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat' and op != 'Eltwise':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=False if op == 'Conv3d' else True, num_segments=num_segments)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            elif op == 'Concat':
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = self._channel_dict[in_var[0]]
                self._channel_dict[out_var[0]] = channel

        model_dict = self.state_dict()
        print("pretrained_parts: ", pretrained_parts)

        rename_layer_dict = {
            'fc_final': 'fc',
            'res5b_bn': 'layer4.1.bn2',
            'res5b_2': 'layer4.1.conv2',
            'res5b_1_bn': 'layer4.1.bn1',
            'res5b_1': 'layer4.1.conv1',
            'res5a_bn': 'layer4.0.bn2',
            'res5a_2': 'layer4.0.conv2',
            'res5a_1_bn': 'layer4.0.bn1',
            'res5a_1': 'layer4.0.conv1',
            'res4b_bn': 'layer3.1.bn2',
            'res4b_2': 'layer3.1.conv2',
            'res4b_1_bn': 'layer3.1.bn1',
            'res4b_1': 'layer3.1.conv1',
            'res4a_bn': 'layer3.0.bn2',
            'res4a_2': 'layer3.0.conv2',
            'res4a_1_bn': 'layer3.0.bn1',
            'res4a_1': 'layer3.0.conv1',
            'res3b_bn': 'layer2.1.bn2',
            'res3b_2': 'layer2.1.conv2',
            'res3b_1_bn': 'layer2.1.bn1',
            'res3b_1': 'layer2.1.conv1',
            'res3a_bn': 'layer2.0.bn2',
            'res3a_2': 'layer2.0.conv2',
            'res3a_1_bn': 'layer2.0.bn1',
            'res3a_1': 'layer2.0.conv1'
        }

        if pretrained_parts == "scratch":
            
            new_state_dict = {}
        
        elif pretrained_parts == "2D":
            
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            new_state_dict = {k: v for k, v in pretrained_dict_2d['state_dict'].items() if k in model_dict}
        
        elif pretrained_parts == "3D":
            
            pretrained_dict_3d = torch.utils.model_zoo.load_url(weight_url_3d)
            new_state_dict = {}
            for k, v in pretrained_dict_3d['state_dict'].items():
                pre_layer_name = k[7:]
                for key, value in rename_layer_dict.items():
                    if value in pre_layer_name:
                        after_layer_name = pre_layer_name.replace(value, key)
                        new_state_dict[after_layer_name] = v
            # 2d Net dim1 output num: 96, first layer in pretrained 3d Net model (res3a_1) dim1 only have 64, so expand it to 96
            new_state_dict['res3a_1.weight'] = torch.cat((new_state_dict['res3a_1.weight'], torch.split(new_state_dict['res3a_1.weight'], 32, 1)[0]), 1)
        
        elif pretrained_parts == "both":
            
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            new_state_dict = {k: v for k, v in pretrained_dict_2d['state_dict'].items() if k in model_dict}
            pretrained_dict_3d = torch.utils.model_zoo.load_url(weight_url_3d)
            for k, v in pretrained_dict_3d['state_dict'].items():
                pre_layer_name = k[7:]
                for key, value in rename_layer_dict.items():
                    if value in pre_layer_name:
                        after_layer_name = pre_layer_name.replace(value, key)
                        new_state_dict[after_layer_name] = v
            # 2d Net dim1 output num: 96, first layer in pretrained 3d Net model (res3a_1) dim1 only have 64, so expand it to 96
            new_state_dict['res3a_1.weight'] = torch.cat((new_state_dict['res3a_1.weight'], torch.split(new_state_dict['res3a_1.weight'], 32, 1)[0]), 1)

        # init the layer names which is not in pretrained model dict
        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        print("un_init_dict_keys: ", un_init_dict_keys)
        print("\n------------------------------------")

        std = 0.001
        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    print("{} init as: 1".format(k))
                    constant(new_state_dict[k], 1)
                else:
                    print("{} init as: xavier".format(k))
                    xavier_uniform(new_state_dict[k])
            elif 'bias' in k:
                print("{} init as: 0".format(k))
                constant(new_state_dict[k], 0)

        print("------------------------------------")

        self.load_state_dict(new_state_dict)
    


        # self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))

    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct' and op[1] != 'Eltwise':
                # first 3d conv layer judge, the last 2d conv layer's output must be transpose from 4d to 5d
                if op[0] == 'res3a_1' or op[0] == 'res3a_down':
                    inception_3c_output = data_dict['inception_3c_double_3x3_1_bn']
                    inception_3c_transpose_output = torch.transpose(inception_3c_output.view((-1, self.num_segments) + inception_3c_output.size()[1:]), 1, 2)
                    data_dict[op[2]] = getattr(self, op[0])(inception_3c_transpose_output)
                else:
                    data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                    # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            elif op[1] == 'Eltwise':
                try:
                    data_dict[op[2]] = torch.add(data_dict[op[-1][0]], 1, data_dict[op[-1][1]])
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
                # x = data_dict[op[-1]]
                # data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        # print output data size in each layers
        # for k in data_dict.keys():
        #     print(k,": ",data_dict[k].size())
        # exit()

        # "self._op_list[-1][2]" represents: last layer's name(e.g. fc_action)
        return data_dict[self._op_list[-1][2]]