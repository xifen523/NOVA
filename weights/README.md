# Weight release interface

No model weights are included in this repository. Each future NOVA weight release must ship a completed copy of `manifest.template.yaml` alongside a PEFT adapter in safetensors format and `geo_components.safetensors`.

Before loading, download the checkpoint to a local directory and verify its published SHA-256. The loader is offline by default, sets `trust_remote_code=False`, validates a supplied checkpoint hash, rejects symlinks and unsupported files, and reports missing, unexpected, shape-mismatched, skipped, and aligned keys. Legacy pickle components require a separate explicit provenance opt-in and are not part of the standard release interface.

After a weight is published, the standard offline invocation is:

```bash
nova-infer --mode tracking \
  --config configs/reproduction/v2x_gd_nova.yaml \
  --model /path/to/verified/base-model \
  --checkpoint /path/to/verified/nova-checkpoint \
  --checkpoint-sha256 <published-checkpoint-tree-sha256> \
  --vocab-alignment verified-overlap \
  --input /path/to/prepared-detections.json \
  --output /path/to/predictions.json
```

The base model revision, model-tree digest, tokenizer digest, checkpoint digest, license, and download URL remain unresolved until the corresponding weight publication. Do not treat placeholder manifest values as release metadata.
