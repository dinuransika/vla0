# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import base64
import io
import json
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from transformers import GenerationConfig

_orig_from_model_config = GenerationConfig.from_model_config

def _safe_from_model_config(model_config):
    try:
        return _orig_from_model_config(model_config)
    except AttributeError as e:
        if "'dict' object has no attribute 'to_dict'" in str(e):
            print("[patch] from_model_config dict workaround, returning default GenerationConfig")
            return GenerationConfig()
        raise

GenerationConfig.from_model_config = _safe_from_model_config

from rv_train.deploy.data_models import So100Base64DataModel
from rv_train.deploy.model_manager import So100ModelManager



rbv_mm = None

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def rgb_from_base64(base64_string: str) -> np.ndarray:
    img = base64.b64decode(base64_string)
    array_bytes = io.BytesIO(img)
    return np.load(array_bytes)


@app.post("/predict_base64")
async def predict(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr
    start_time = time.time()
    with torch.no_grad():
        output, _ = rbv_mm.forward(image_data, state_data, instr_data)
    print(f"Time taken: {time.time() - start_time}")
    return output


@app.post("/predict_base64_stream")
async def predict_base64_stream(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr

    def generate():
        start_time = time.time()
        assert rbv_mm.cfg.EXP.MODEL == "qwen"
        last_action_txt = ""
        for i in range(rbv_mm.cfg.MODEL.QWEN.horizon):
            with torch.no_grad():
                output, last_action_txt = rbv_mm.forward(
                    image_data,
                    state_data,
                    instr_data,
                    get_one_step_action=True,
                    last_action_txt=last_action_txt,
                )
                print(last_action_txt)
            print(f"Time taken: {time.time() - start_time}")
            yield json.dumps({"index": i, "value": output}) + "\n"
        yield json.dumps({"time_taken": time.time() - start_time}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def get_ip_address():
    import socket

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


if __name__ == "__main__":
    rbv_mm = So100ModelManager()
    import torch.nn as nn

    RS = {"type": "default", "rope_type": "default", "mrope_section": [16, 24, 24]}

    # Patch all configs that have a rope_scaling attribute
    def patch_configs(obj, seen=None):
        if seen is None: seen = set()
        if id(obj) in seen: return
        seen.add(id(obj))
        if hasattr(obj, "rope_scaling"):
            obj.rope_scaling = RS
        for name in ("text_config", "vision_config", "config"):
            sub = getattr(obj, name, None)
            if sub is not None:
                patch_configs(sub, seen)

    patch_configs(rbv_mm.model.model.config)

    # Patch every attention module's rope_scaling attribute
    n_patched = 0
    for module in rbv_mm.model.model.modules():
        if hasattr(module, "rope_scaling"):
            module.rope_scaling = RS
            n_patched += 1
    print(f"patched rope_scaling on {n_patched} modules")
    import torch
    from safetensors.torch import load_file

    sd_path = "/Users/dinura.dissanayake/Desktop/vla0/checkpoint-4000/model.safetensors"
    sd = load_file(sd_path)
    print("lm_head.weight in safetensors:", "lm_head.weight" in sd)
    print("keys matching lm_head/embed:", [k for k in sd.keys() if "lm_head" in k or "embed_tokens" in k])

    cfg = rbv_mm.model.model.config
    print("tie_word_embeddings:", getattr(cfg, "tie_word_embeddings", None))
    if hasattr(cfg, "text_config"):
        print("text_config.tie_word_embeddings:", getattr(cfg.text_config, "tie_word_embeddings", None))

    emb = rbv_mm.model.model.get_input_embeddings().weight
    head = rbv_mm.model.model.get_output_embeddings().weight
    print("embed shape:", emb.shape, "lm_head shape:", head.shape)
    print("tied (same memory):", emb.data_ptr() == head.data_ptr())
    print("equal values:", torch.allclose(emb[:10], head[:10]))
    rbv_mm.model.model.tie_weights()
    # verify
    emb = rbv_mm.model.model.get_input_embeddings().weight
    head = rbv_mm.model.model.get_output_embeddings().weight
    print("tied after fix:", emb.data_ptr() == head.data_ptr())
    PORT = 10000
    print()
    print(f"IP address: {get_ip_address()}")
    print(f"Go to http://{get_ip_address()}:{PORT}/docs for the API documentation")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
