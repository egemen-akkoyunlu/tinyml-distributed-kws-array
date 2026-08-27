import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Subset
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from datasets import SpeechCommandsDataset

# Try importing esp_ppq APIs
try:
    from esp_ppq.api import espdl_quantize_torch, QuantizationSettingFactory
    import esp_ppq.parser.espdl_exporter as espdl_exporter_module

    # Runtime patch to ensure all hardware shift exponents are >= 0 (prevents ESP32-S3 SIMD overflow)
    def patch_espdl_exporter():
        orig_prepare_graph = espdl_exporter_module.EspdlExporter.prepare_graph
        def safe_prepare_graph(self, graph, *args, **kwargs):
            graph = orig_prepare_graph(self, graph, *args, **kwargs)
            info = espdl_exporter_module.ExporterPatternInfo()
            for op in graph.topological_sort():
                if op.type in ["Conv", "Gemm", "MatMul", "Mul"]:
                    in0 = info.get_var_exponents(op.inputs[0].name)
                    in1 = info.get_var_exponents(op.inputs[1].name)
                    out = info.get_var_exponents(op.outputs[0].name)
                    if in0 and in1 and out:
                        req = in0[0] + in1[0]
                        if out[0] < req:
                            print(f"[Auto-Fix] Clamped exponent for {op.name} ({op.type}): {out[0]} -> {req} (shift >= 0 satisfied)")
                            info.add_var_exponents(op.outputs[0].name, req)
            return graph
        espdl_exporter_module.EspdlExporter.prepare_graph = safe_prepare_graph

    patch_espdl_exporter()

except ImportError:
    print("Error: esp-ppq is not installed or accessible in this Python environment.")
    print("Please run this script inside kws_env: /home/acar/kws_env/bin/python3 quantize_to_espdl.py")
    exit(1)


class SpectrogramTransform:
    def __init__(self, n_fft=256, hop_length=160, n_mels=40):
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            mel_scale="htk",
            norm=None,
            center=False
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)
        return spec


def main():
    print("==================================================")
    print("  Post-Training Quantization (PTQ) for ESP32-S3")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computation device: {device}")
    
    # Fix deterministic random seeds for stable PTQ calibration
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    target = "esp32s3"
    num_of_bits = 8
    
    # Path to experiment config and weights
    import sys
    import glob

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        if not os.path.exists(os.path.join(log_dir, ".hydra", "config.yaml")):
            sub_dirs = sorted(glob.glob(os.path.join(log_dir, "*/")), key=os.path.getmtime)
            if sub_dirs:
                log_dir = sub_dirs[-1].rstrip("/")
        print(f"Using specified log directory: {log_dir}")
    else:
        # Automatically find the latest run folder under logs/
        log_dirs = sorted(glob.glob("logs/*/*/"), key=os.path.getmtime)
        if not log_dirs:
            log_dirs = sorted(glob.glob("logs/*/"), key=os.path.getmtime)
        if log_dirs:
            log_dir = log_dirs[-1].rstrip("/")
            print(f"No log directory specified. Automatically detected latest run: {log_dir}")
        else:
            log_dir = "logs/export_nfft_128_all_classes/2026-07-08_11-23-36"
            print(f"No log directory detected. Falling back to default: {log_dir}")

    config_path = os.path.join(log_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    allowed_classes = cfg.dataset.allowed_classes
    num_classes = len(allowed_classes)
    print(f"Loaded config for {num_classes} classes: {allowed_classes}")

    # Resolve weights path dynamically using the experiment_name from config
    experiment_name = cfg.get("experiment_name", "model")
    model_weights_path = os.path.join(log_dir, f"best_{experiment_name}.pth")

    # Instantiate model in export mode
    model = instantiate(cfg.model, num_classes=num_classes, export_mode=True).to(device)
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=device)
        # Check for output layer class mismatch and slice weights for the allowed classes
        if 'fc.weight' in state_dict and state_dict['fc.weight'].shape[0] != num_classes:
            trained_classes = state_dict['fc.weight'].shape[0]
            print(f"Detecting class size mismatch: trained model has {trained_classes} classes, target has {num_classes} classes.")
            print(f"Slicing fc.weight and fc.bias to the first {num_classes} allowed classes...")
            state_dict['fc.weight'] = state_dict['fc.weight'][:num_classes, :]
            state_dict['fc.bias'] = state_dict['fc.bias'][:num_classes]
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {model_weights_path} with strict=False success!")
    else:
        print(f"Warning: Weights not found at {model_weights_path}, using random weights for testing.")
    model.eval()

    # Create REAL Calibration DataLoader from training speech commands
    n_mel_bins = cfg.model.n_mel_bins
    time_frames = 100
    input_shape = [1, n_mel_bins, time_frames]
    
    print("Preparing real calibration dataset (128 speech command samples)...")
    spec_transform = SpectrogramTransform(
        n_fft=cfg.model.get('n_fft', 256),
        hop_length=cfg.model.get('hop_length', 160),
        n_mels=n_mel_bins
    )
    train_dataset = SpeechCommandsDataset(
        root_dir="speech_commands_dataset",
        transform=spec_transform,
        allowed_classes=allowed_classes,
        subset="training",
        augment=False
    )
    calib_indices = []
    rng = np.random.RandomState(42)
    samples_per_class = max(8, 256 // len(allowed_classes))
    for c_idx, c_name in enumerate(allowed_classes):
        c_matches = [i for i, label in enumerate(train_dataset.labels) if label == c_idx]
        if c_matches:
            selected = rng.choice(c_matches, min(samples_per_class, len(c_matches)), replace=False)
            calib_indices.extend(selected)
            
    calib_dataset = Subset(train_dataset, calib_indices)
    calib_steps = len(calib_dataset)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
    print(f"Calibration dataset prepared: {calib_steps} total balanced samples ({samples_per_class} per class).")

    def calib_collate_fn(batch):
        specs, _ = batch
        if specs.dim() == 4 and specs.size(1) == 1:
            specs = specs.squeeze(1)
        # Apply mean-std normalization to match training and C++ runtime
        mean = specs.mean(dim=(1, 2), keepdim=True)
        std = specs.std(dim=(1, 2), keepdim=True) + 1e-5
        specs = (specs - mean) / std
        return specs.to(device)

    # Output paths
    out_dir = "zephyr_kws/src"
    os.makedirs(out_dir, exist_ok=True)
    espdl_model_path = os.path.join(out_dir, "model.espdl")

    # Configure Cross-Layer Equalization and MinMax activation calibration
    setting = QuantizationSettingFactory.espdl_setting()
    setting.equalization = True
    setting.quantize_activation_setting.calib_algorithm = 'minmax'

    print(f"Running INT8 PTQ with esp-ppq for target '{target}'...")
    
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=espdl_model_path,
        calib_dataloader=calib_dataloader,
        calib_steps=calib_steps,
        input_shape=input_shape,
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=calib_collate_fn,
        setting=setting,
        device=str(device),
        error_report=True,
        skip_export=False,
        verbose=1,
    )

    print("==================================================")
    print(f"SUCCESS: Quantized model exported to {espdl_model_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
