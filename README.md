# DrakoMedia LTX-2.3 · RunPod Serverless

Worker privado de vídeo para DrakoMedia Studio Pro Cloud.

## Diseño

- Pipeline oficial `DistilledPipeline` de Lightricks.
- Checkpoint oficial FP8 destilado, fijado por revisión.
- Ocho pasos en la primera etapa y cuatro en el refinado.
- El checkpoint grande se sirve desde la caché de modelos de RunPod.
- Gemma 3 QAT y el upscaler oficial están fijados dentro de la imagen.
- Active workers `0`, max workers `1` y FlashBoot activado.
- Los archivos viajan por enlaces R2 firmados con caducidad corta.
- No hay claves de Cloudflare dentro del worker.
- Solo se escriben objetos bajo `safe-to-delete/`.

## Endpoint recomendado

- Tipo: Queue.
- Caché de modelo: `https://huggingface.co/Lightricks/LTX-2.3-fp8`.
- GPU: H100 80 GB.
- Active workers: 0.
- Max workers: 1.
- Idle timeout: 5 segundos.
- Execution timeout: 1800 segundos.
- FlashBoot: activado.
