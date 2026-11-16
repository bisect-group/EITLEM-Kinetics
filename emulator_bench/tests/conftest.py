def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: CUDA tolerance tests (side-stream loader, TF32 and cudnn.benchmark)")
    config.addinivalue_line("markers", "esm_weights: needs the real ESM1v weights in the torch hub cache")
