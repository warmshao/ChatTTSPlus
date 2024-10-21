import pdb
import time
import numpy as np
import torch
from torch.cuda import nvtx
from .base_model import BaseModel
from .predictor import numpy_to_torch_dtype_dict


class LlamaTRTModel(BaseModel):
    """
    Llama TensorRT Model
    """

    def __init__(self, **kwargs):
        super(LlamaTRTModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        for i, inp in enumerate(self.predictor.inputs):
            if inp["name"] == "past_key_values":
                self.kv_cache_dtype = numpy_to_torch_dtype_dict[inp['dtype']]
                self.kv_cache_shape = inp['shape']

    def create_kv_cache(self, batch_size=1):
        self.kv_caches = torch.empty(self.kv_cache_shape[0], self.kv_cache_shape[1], batch_size, self.kv_cache_shape[3],
                                     self.max_seq_len + 1, self.kv_cache_shape[5]).to(self.device,
                                                                                      dtype=self.kv_cache_dtype)
        self.kv_ind = 1

    def clear_kv_caches(self):
        self.kv_ind = 1

    def get_cur_kv_caches(self):
        return self.kv_caches[:, :, :, :, 1:self.kv_ind]

    def input_process(self, *data, **kwargs):
        return data

    def output_process(self, *data, **kwargs):
        return data[0]

    def predict_trt(self, *data, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {}
        cur_input_shape = None
        for i, inp in enumerate(self.predictor.inputs):
            if inp["name"] != "past_key_values":
                if inp["name"] == "inputs_embeds":
                    cur_input_shape = data[i].shape
                if isinstance(data[i], torch.Tensor):
                    feed_dict[inp['name']] = data[i].to(device=self.device,
                                                        dtype=numpy_to_torch_dtype_dict[inp['dtype']])
                else:
                    feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                          dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            else:
                feed_dict[inp['name']] = self.kv_caches[:, :, :, :, :self.kv_ind]
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            if out["name"] == "cur_key_values":
                out_shape = self.kv_cache_shape[:]
                out_shape[2] = cur_input_shape[0]
                out_shape[4] = cur_input_shape[1]
                out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
                new_kv_len = out_tensor.shape[4]
                self.kv_caches[:, :, :, :, self.kv_ind:self.kv_ind + new_kv_len] = out_tensor.clone()
                self.kv_ind += new_kv_len
            else:
                out_shape = cur_input_shape[:]
                out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
                outs.append(out_tensor)
        nvtx.range_pop()
        return outs

    def predict(self, *data, **kwargs):
        data = self.input_process(*data, **kwargs)
        preds = self.predict_trt(*data, **kwargs)
        outputs = self.output_process(*preds, **kwargs)
        return outputs
