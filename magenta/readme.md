# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt

# Magenta 모델 다운로드 스크립트
cat > download_models.py << 'EOL'
import os
import urllib.request

# 체크포인트 디렉토리 생성
os.makedirs("checkpoints", exist_ok=True)

# 모델 체크포인트 다운로드
print("Music VAE 체크포인트 다운로드 중...")
urllib.request.urlretrieve(
    "https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar",
    "checkpoints/cat-mel_2bar_big.tar"
)
print("다운로드 완료!")
EOL

# 모델 다운로드 실행
python download_models.py

# Ollama 설치 및 Llama 3.2 모델 다운로드
# 참고: Ollama는 별도로 설치해야 합니다
# Linux/macOS 설치: curl -fsSL https://ollama.com/install.sh | sh
# Windows 설치: Ollama 웹사이트에서 설치 프로그램 다운로드

# Llama 3.2 모델 다운로드 (Ollama가 이미 설치되어 있다고 가정)
ollama pull llama3.2:latest

# 서버 실행
python main.py