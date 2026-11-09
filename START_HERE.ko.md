# BAGEL GPU 실행 가이드

## 1. GPU 서버 준비

`inference.py`가 포함된 이 저장소를 Linux GPU 서버에 클론하고, 저장소 루트에서 다음 명령을 실행합니다. NVIDIA 드라이버와 BF16을 지원하는 CUDA GPU가 필요합니다. FlashAttention을 소스에서 빌드할 경우 PyTorch와 호환되는 CUDA 툴킷(`nvcc`)과 C++ 컴파일러가 필요합니다.

```bash
nvidia-smi
nvcc --version
df -h .

conda create -n bagel python=3.10 -y
conda activate bagel
python -m pip install -r requirements.txt
python -m pip install flash_attn==2.5.8 --no-build-isolation
python -m pip check
```

CUDA, BF16과 FlashAttention을 확인합니다.

```bash
python - <<'PY'
import torch
import flash_attn

assert torch.cuda.is_available(), "CUDA GPU is unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
```

Apple Silicon Mac에서는 `python3 inference.py --help`와 인자 처리까지 확인할 수 있습니다. 실제 추론 경로는 CUDA autocast와 FlashAttention에 의존합니다.

## 2. 공식 가중치 다운로드

가중치 파일만 약 30GB입니다. Python 환경과 다운로드 임시 파일, 오프로딩 파일을 위한 디스크 여유 공간도 확보합니다. [모델 파일 목록](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/tree/main)

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    local_dir="models/BAGEL-7B-MoT",
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
PY
```

## 3. CLI 실행

[한국어 README의 개별 예제](README.ko.md#실행-방법)를 실행하거나, 세 작업을 한 번씩 실행합니다.

```bash
bash bash-code-example.txt
```

입력 이미지는 명령을 실행하는 서버의 파일 경로를 지정합니다. 결과는 `results/`에 저장됩니다.

| 옵션 | 동작 |
| --- | --- |
| `--model-path` | 가중치 폴더. 기본값은 `models/BAGEL-7B-MoT` |
| `--seed`, `--steps` | 기본값은 각각 `42`, `50`. 스텝 수는 이미지 생성과 편집에 적용 |
| `--cfg-text-scale`, `--cfg-image-scale` | 텍스트·이미지 조건의 강도. 기본값은 텍스트 `4.0`, 이미지 생성 `1.0`, 이미지 편집 `2.0` |
| `--think`, `--max-tokens` | 생성·편집 시 추론 텍스트 출력 옵션. 토큰 한도의 기본값은 `1000`이며 이미지 이해 응답에도 적용 |
| `--max-mem-per-gpu` | GPU별 모델 배치 한도. 기본값은 `40GiB` |

GPU 메모리에는 모델 가중치 외에도 실제 추론에 필요한 공간이 들어갑니다. `--max-mem-per-gpu`는 모델 배치에 적용되는 값이므로 GPU 전체 메모리보다 여유 있게 잡습니다. 기록된 실험 환경의 후보인 NVIDIA L40S는 VRAM 48GB 모델입니다. [NVIDIA L40S 사양](https://www.nvidia.com/en-gb/data-center/l40s/)

## 4. 공식 Gradio 앱 사용

CLI는 BF16으로 로드합니다. 공식 웹 UI에서는 일반 모드와 NF4 양자화 모드를 사용할 수 있습니다.

```bash
# 일반 모드
python app.py --model_path models/BAGEL-7B-MoT --server_name 127.0.0.1 --server_port 7860

# NF4 모드
python app.py --model_path models/BAGEL-7B-MoT --mode 2 --server_name 127.0.0.1 --server_port 7860
```

Mac에서 아래처럼 SSH 터널을 열고 브라우저로 `http://127.0.0.1:7860`에 접속합니다. `user@host`는 GPU 서버의 SSH 접속 대상으로 바꾸고, 별도 SSH 포트를 쓰는 서버라면 `-p`로 지정합니다.

```bash
ssh -N -L 7860:127.0.0.1:7860 user@host
```

## 코드 읽는 순서

1. [inference.py](inference.py)에서 CLI 인자 처리와 작업별 호출을 보고, [inference.ipynb](inference.ipynb)에서 원본 로딩 과정을 확인합니다.
2. [inferencer.py](inferencer.py)의 `InterleaveInferencer`가 텍스트와 이미지 문맥을 쌓아 출력 작업을 호출합니다.
3. [bagel.py](modeling/bagel/bagel.py)의 `generate_text`와 `generate_image`가 각 출력의 진입점입니다.
4. [qwen2_navit.py](modeling/bagel/qwen2_navit.py)는 언어 모델과 MoT 디코더, [siglip_navit.py](modeling/bagel/siglip_navit.py)는 시각 인코더, [autoencoder.py](modeling/autoencoder.py)는 이미지 잠재 표현의 인코딩·디코딩을 맡습니다.
