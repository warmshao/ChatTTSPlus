import pdb
import time

import torch
from torch.cuda import nvtx
from .base_model import BaseModel
from .predictor import numpy_to_torch_dtype_dict


class LlamaModel(BaseModel):
    """
    Llama  Model
    """

    def __init__(self, **kwargs):
        super(LlamaModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        for i, inp in enumerate(self.predictor.inputs):
            if inp["name"] == "past_key_values":
                self.kv_cache_dtype = inp['dtype']
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

    def predict(self, *data, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if inp["name"] != "past_key_values":
                if isinstance(data[i], torch.Tensor):
                    feed_dict[inp['name']] = data[i].to(numpy_to_torch_dtype_dict[inp['dtype']])
                else:
                    feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                          dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            else:
                feed_dict[inp['name']] = self.kv_caches[:, :, :, :, :self.kv_ind]
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            output_shapes = kwargs.get("output_shapes", {})
            if out["name"] in output_shapes:
                out_shape = output_shapes[out["name"]]
            else:
                out_shape = out["shape"]
            out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
            if out["name"] == "cur_key_values":
                new_kv_len = out_tensor.shape[4]
                self.kv_caches[:, :, :, :, self.kv_ind:self.kv_ind + new_kv_len] = out_tensor.clone()
                self.kv_ind += new_kv_shape
            else:
                outs.append(out_tensor.clone())
        nvtx.range_pop()
        return outs

    def predict(self, *data, **kwargs):
        data = self.input_process(*data)
        preds = self.predict(*data)
        outputs = self.output_process(*preds)
        return outputs
