# Rapid-MLX-panel

Lokal kontrollpanel för `rapid-mlx serve`. Den lyssnar bara på `127.0.0.1:8770`.

```bash
python3 /Users/otto/LLM/rapid-mlx-panel/panel.py
```

Platser, portar, API-namn, vikter och serve-flaggor sparas i `~/.config/rapid-mlx-panel/slots.json`. Loggar för processer som panelen själv startat ligger i `~/.config/rapid-mlx-panel/logs/`.

En server som redan körs kopplas till platsen med samma port och visas som startad utanför panelen. Den stoppas bara när du väljer Stoppa eller Starta om.

KV-cache kan vara standard, bf16, int8, int4, TurboQuant v4, TurboQuant k8v4 eller TurboQuant av. MTP-utkast och MTP-djup kan stå på katalogens val. Under fälten står vad standard betyder för den valda vikten. Ett vision-körfält kan inte sparas tillsammans med MTP. Befintliga platser behåller ett eget MTP-djup tills du växlar till katalogens djup.

Motorn kan vara rapid-mlx, TensorFold Zig (`tensorfold-native` på `127.0.0.1`) eller ComfyUI. ComfyUI startar diffusionsservern med `--listen 127.0.0.1`. Vikterna väljs i arbetsflödet, inte vid processstart. `COMFYUI_PYTHON` och `COMFYUI_ROOT` kan peka om tolk och installation. Zig-platsen har drafter, Cortheon, learn och slide. Learn sparar prompt-cachen. Slide skriver om vikterna i checkpointen. DFlash2-repot `z-lab/Qwen3.8-27B-DFlash2` går inte att köra som drafter i Zig 1.1.1.
