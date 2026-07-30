from huggingface_hub import hf_hub_download, snapshot_download


LTX_REVISION = "4229404625088d21c4f112eb640fb04a0900ee25"
GEMMA_REVISION = "d62fe4f1995ade703b49a0f3c0d0f161237ef437"
hf_hub_download(
    "Lightricks/LTX-2.3",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    revision=LTX_REVISION,
    local_dir="/models/ltx-2.3",
)
snapshot_download(
    "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
    revision=GEMMA_REVISION,
    local_dir="/models/gemma-3-12b",
    ignore_patterns=["README.md", ".gitattributes"],
)
