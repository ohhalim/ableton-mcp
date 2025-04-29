# main.py - FastAPI 백엔드 구현
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import uuid
import pretty_midi
import faiss
import pickle
import subprocess
import json
import asyncio
from datetime import datetime  # 누락된 임포트 추가
from typing import List, Dict, Any, Optional
import requests  # 모델 다운로드를 위한 임포트 추가

# Magenta 관련 임포트
import note_seq
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

# Ollama 클라이언트
import httpx

app = FastAPI(title="MIDI RAG 시스템")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 데이터베이스 경로
MODEL_CHECKPOINT_DIR = "checkpoints"
DATABASE_DIR = "database"
MIDI_STORAGE_DIR = "midi_files"

# 디렉토리 생성
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(MIDI_STORAGE_DIR, exist_ok=True)

# 데이터베이스 파일 경로
FAISS_INDEX_PATH = os.path.join(DATABASE_DIR, "midi_index.faiss")
MIDI_METADATA_PATH = os.path.join(DATABASE_DIR, "midi_metadata.pkl")

# 글로벌 변수
midi_metadata = {}
index = None
music_vae = None

# Music VAE 모델 설정
MUSIC_VAE_CONFIG = configs.CONFIG_MAP.get('cat-mel_2bar_big')
MUSIC_VAE_CHECKPOINT = os.path.join(MODEL_CHECKPOINT_DIR, "cat-mel_2bar_big.tar")
# 모델 URL (GitHub에서 호스팅된 모델 체크포인트)
MUSIC_VAE_MODEL_URL = "https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar"

# Ollama API 설정
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# 모델 초기화 함수
def initialize_models():
    global music_vae, index, midi_metadata
    
    # Music VAE 모델 체크포인트 다운로드 (필요한 경우)
    if not os.path.exists(MUSIC_VAE_CHECKPOINT):
        try:
            print(f"Music VAE 모델 체크포인트 다운로드 중... ({MUSIC_VAE_MODEL_URL})")
            response = requests.get(MUSIC_VAE_MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MUSIC_VAE_CHECKPOINT, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Music VAE 모델 체크포인트가 다운로드되었습니다.")
        except Exception as e:
            print(f"모델 체크포인트 다운로드 중 오류 발생: {e}")
    
    # Music VAE 모델 로드
    try:
        music_vae = TrainedModel(
            MUSIC_VAE_CONFIG, 
            batch_size=4, 
            checkpoint_dir_or_path=MUSIC_VAE_CHECKPOINT
        )
        print("Music VAE 모델이 로드되었습니다.")
    except Exception as e:
        print(f"Music VAE 모델 로드 중 오류 발생: {e}")
        music_vae = None
    
    # FAISS 인덱스 로드
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"FAISS 인덱스가 로드되었습니다. 인덱스 크기: {index.ntotal}")
        except Exception as e:
            print(f"FAISS 인덱스 로드 중 오류 발생: {e}")
            index = faiss.IndexFlatL2(512)  # 오류 시 새 인덱스 생성
            print("새 FAISS 인덱스가 생성되었습니다 (기존 인덱스 로드 실패로 인한 생성).")
    else:
        # 새 인덱스 생성 (512 차원 벡터 사용)
        index = faiss.IndexFlatL2(512)
        print("새 FAISS 인덱스가 생성되었습니다.")
    
    # MIDI 메타데이터 로드
    if os.path.exists(MIDI_METADATA_PATH):
        try:
            with open(MIDI_METADATA_PATH, 'rb') as f:
                midi_metadata = pickle.load(f)
            print(f"MIDI 메타데이터가 로드되었습니다. 항목 수: {len(midi_metadata)}")
        except Exception as e:
            print(f"MIDI 메타데이터 로드 중 오류 발생: {e}")
            midi_metadata = {}
    else:
        midi_metadata = {}
        print("새 MIDI 메타데이터 사전이 생성되었습니다.")

# MIDI 벡터화 함수
def vectorize_midi(midi_data):
    """MIDI 데이터를 벡터로 변환"""
    try:
        # pretty_midi 객체로 변환
        pm = pretty_midi.PrettyMIDI(midi_data=midi_data)
        
        # 화성 추출
        chords = extract_chords(pm)
        
        # 메타 정보 추출 (템포, 키, 박자, 코드 밀도 등)
        meta_features = extract_meta_features(pm)
        
        # 피치 분포 추출
        pitch_hist = extract_pitch_histogram(pm)
        
        # 리듬 패턴 추출
        rhythm_pattern = extract_rhythm_pattern(pm) 
        
        # 모든 특성을 하나의 벡터로 합침
        # (여기서는 간단한 예시로, 실제로는 더 복잡한 처리가 필요할 수 있음)
        features = np.concatenate([
            chords.flatten(),
            meta_features,
            pitch_hist,
            rhythm_pattern
        ])
        
        # 벡터 크기 맞추기 (512 차원으로 패딩 또는 잘라내기)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        else:
            features = features[:512]
        
        return features.astype(np.float32)
    except Exception as e:
        print(f"MIDI 벡터화 중 오류 발생: {e}")
        # 오류 시 임의의 벡터 반환
        return np.random.rand(512).astype(np.float32)

def extract_chords(pm):
    """MIDI 파일에서 화성(코드) 추출"""
    # 간단한 구현: 각 타임스텝마다 동시에 연주되는 노트의 조합을 추출
    # 실제 구현에서는 더 복잡한 화성 분석이 필요할 수 있음
    chords = np.zeros((12, 8))  # 12개 피치 클래스 × 8개 타임 슬롯
    
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                # 노트 시작 시간을 8개 타임 슬롯으로 양자화
                time_idx = min(int(note.start / pm.get_end_time() * 8), 7)
                # 피치 클래스 (0-11)
                pitch_class = note.pitch % 12
                chords[pitch_class, time_idx] = 1
    
    return chords

def extract_meta_features(pm):
    """템포, 키, 박자 등의 메타 정보 추출"""
    # 간단한 구현으로 7개의 메타 특성 추출
    meta = np.zeros(7)
    
    # 템포 (평균 BPM)
    if len(pm.get_tempo_changes()) > 0:
        meta[0] = np.mean(pm.get_tempo_changes()[1]) / 180.0  # 정규화
    
    # 악곡 길이
    meta[1] = min(pm.get_end_time() / 60.0, 1.0)  # 최대 1분으로 정규화
    
    # 피아노 롤 밀도
    total_notes = sum(len(instrument.notes) for instrument in pm.instruments)
    meta[2] = min(total_notes / 500.0, 1.0)  # 최대 500개 노트로 정규화
    
    # 평균 노트 길이    
    if total_notes > 0:
        avg_duration = np.mean([note.end - note.start for instrument in pm.instruments 
                              for note in instrument.notes])
        meta[3] = min(avg_duration / 2.0, 1.0)  # 최대 2초로 정규화
    
    # 평균 음높이 (MIDI 노트 번호)
    if total_notes > 0:
        avg_pitch = np.mean([note.pitch for instrument in pm.instruments 
                           for note in instrument.notes])
        meta[4] = (avg_pitch - 36) / (96 - 36)  # 36-96 범위로 정규화
    
    # 노트 다양성 (사용된 피치 클래스 수)
    used_pitches = set()
    for instrument in pm.instruments:
        for note in instrument.notes:
            used_pitches.add(note.pitch % 12)
    meta[5] = len(used_pitches) / 12.0
    
    # 악기 수
    meta[6] = min(len(pm.instruments) / 8.0, 1.0)  # 최대 8개 악기로 정규화
    
    return meta

def extract_pitch_histogram(pm):
    """피치 분포 추출"""
    # 12개 피치 클래스의 히스토그램
    pitch_hist = np.zeros(12)
    
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_class = note.pitch % 12
                pitch_hist[pitch_class] += 1
    
    # 정규화
    if np.sum(pitch_hist) > 0:
        pitch_hist = pitch_hist / np.sum(pitch_hist)
    
    return pitch_hist

def extract_rhythm_pattern(pm):
    """리듬 패턴 추출"""
    # 16개 타임 슬롯의 노트 온셋 패턴
    rhythm = np.zeros(16)
    
    # 악곡 길이
    song_length = pm.get_end_time()
    
    for instrument in pm.instruments:
        for note in instrument.notes:
            # 노트 시작 시간을 16개 타임 슬롯으로 양자화
            time_idx = min(int(note.start / song_length * 16), 15)
            rhythm[time_idx] += 1
    
    # 정규화
    if np.sum(rhythm) > 0:
        rhythm = rhythm / np.max(rhythm)
    
    return rhythm

# 입력 모델 정의
class HarmonyInput(BaseModel):
    midi_data: str  # Base64 인코딩된 MIDI 데이터
    description: Optional[str] = None  # 생성에 대한 추가 설명 (옵션)
    temperature: float = 0.5  # 생성 다양성
    num_variations: int = 3  # 생성할 변형 수

# 응답 모델 정의
class MIDIResponse(BaseModel):
    generated_midis: List[str]  # Base64 인코딩된 생성된 MIDI 파일들
    message: str

# 상태 확인 엔드포인트
@app.get("/api/status")
async def check_status():
    """시스템 상태 확인"""
    return {
        "status": "online",
        "midi_files_count": len(midi_metadata),
        "music_vae_loaded": music_vae is not None,
        "faiss_index_size": index.ntotal if index else 0
    }

# Ollama로 LLM 쿼리 함수
async def query_llm(prompt, context=None):
    """Ollama API를 통해 LLM에 쿼리"""
    try:
        message = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        if context:
            message["context"] = context
        
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_API_URL, json=message, timeout=30.0)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Ollama API 오류: {response.status_code} - {response.text}")
                return "LLM을 쿼리하는 동안 오류가 발생했습니다." 
    except Exception as e:
        print(f"Ollama 쿼리 중 오류 발생: {e}")
        return "LLM을 쿼리하는 동안 오류가 발생했습니다."

# Magenta를 통한 MIDI 생성 함수
def generate_midi_with_magenta(chord_sequence, similar_midis, description=None, temperature=0.5):
    """Magenta를 사용하여 MIDI 생성"""
    try:
        # 유사한 MIDI 파일들의 스타일 요소 추출
        z_vectors = []
        for midi_id in similar_midis:
            if midi_id in midi_metadata:
                midi_path = midi_metadata[midi_id]["path"]
                try:
                    # MIDI 파일 로드
                    pm = pretty_midi.PrettyMIDI(midi_path)
                    sequence = note_seq.midi_to_note_sequence(pm)
                    
                    # 시퀀스를 2마디 세그먼트로 분할
                    segments = mm.extract_subsequences(sequence, 2.0)
                    
                    # 각 세그먼트에 대해 Z 벡터 인코딩
                    for segment in segments[:2]:  # 처음 2개 세그먼트만 사용
                        try:
                            z = music_vae.encode([segment])[0]
                            z_vectors.append(z)
                        except Exception as e:
                            print(f"세그먼트 인코딩 오류: {e}")
                            continue
                except Exception as e:
                    print(f"MIDI 파일 처리 오류 ({midi_path}): {e}")
                    continue
        
        # 입력 코드 시퀀스를 NoteSequence로 변환
        input_sequence = note_seq.protobuf.music_pb2.NoteSequence()
        # 여기서 chord_sequence에서 화성 정보를 추출하여 입력 시퀀스에 추가
        
        # Z 벡터 보간 및 새 시퀀스 생성
        generated_sequences = []
        
        # 충분한 Z 벡터가 있으면 보간 사용
        if len(z_vectors) >= 2:
            for i in range(4):  # 4개 생성
                # 랜덤하게 두 벡터 선택하여 보간
                idx1, idx2 = np.random.choice(len(z_vectors), 2, replace=False)
                interp_ratio = np.random.uniform(0.2, 0.8)
                z_interp = z_vectors[idx1] * interp_ratio + z_vectors[idx2] * (1 - interp_ratio)
                
                # 온도 조정하여 생성
                sequences = music_vae.decode([z_interp], temperature=temperature)
                if sequences:
                    generated_sequences.append(sequences[0])
        else:
            # Z 벡터가 부족하면 샘플링 사용
            for i in range(4):
                try:
                    sequences = music_vae.sample(n=1, temperature=temperature)
                    if sequences:
                        generated_sequences.append(sequences[0])
                except Exception as e:
                    print(f"샘플링 오류: {e}")
        
        # NoteSequence를 MIDI로 변환
        generated_midis = []
        for i, sequence in enumerate(generated_sequences):
            output_file = os.path.join(MIDI_STORAGE_DIR, f"generated_{uuid.uuid4()}.mid")
            try:
                pm = note_seq.note_sequence_to_pretty_midi(sequence)
                pm.write(output_file)
                
                # MIDI 파일을 바이트로 읽기
                with open(output_file, 'rb') as f:
                    midi_bytes = f.read()
                
                # Base64로 인코딩
                import base64
                midi_b64 = base64.b64encode(midi_bytes).decode('utf-8')
                generated_midis.append(midi_b64)
            except Exception as e:
                print(f"MIDI 변환 오류: {e}")
        
        return generated_midis
    except Exception as e:
        print(f"Magenta MIDI 생성 오류: {e}")
        return []

# API 엔드포인트 정의
@app.post("/api/upload-midi")
async def upload_midi(file: UploadFile = File(...), description: str = None):
    """사용자의 MIDI 파일을 업로드하고 데이터베이스에 저장"""
    try:
        # 파일 읽기
        content = await file.read()
        
        # 파일 저장
        file_id = str(uuid.uuid4())
        file_path = os.path.join(MIDI_STORAGE_DIR, f"{file_id}.mid")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # MIDI 벡터화
        vector = vectorize_midi(content)
        
        # FAISS 인덱스에 추가
        if index is not None:
            index.add(np.array([vector]))
            
            # 메타데이터 업데이트
            midi_metadata[file_id] = {
                "path": file_path,
                "filename": file.filename,
                "description": description,
                "upload_date": str(datetime.now())
            }
            
            # 데이터베이스 저장
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(MIDI_METADATA_PATH, 'wb') as f:
                pickle.dump(midi_metadata, f)
            
            return {"message": "MIDI 파일이 성공적으로 업로드되었습니다.", "file_id": file_id}
        else:
            raise HTTPException(status_code=500, detail="FAISS 인덱스가 초기화되지 않았습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 중 오류 발생: {str(e)}")

@app.post("/api/generate-midi")
async def generate_midi(input_data: HarmonyInput):
    """화성 정보를 기반으로 MIDI 생성"""
    try:
        import base64
        # Base64 디코딩
        midi_bytes = base64.b64decode(input_data.midi_data)
        
        # MIDI 벡터화
        query_vector = vectorize_midi(midi_bytes)
        
        # 유사한 MIDI 검색
        if index is not None:
            # FAISS 검색 수행
            k = min(5, index.ntotal)  # 최대 5개 또는 인덱스 크기
            if k > 0:
                distances, indices = index.search(np.array([query_vector]), k)
                
                # 결과 MIDI 파일 ID 가져오기
                similar_midi_ids = []
                for idx in indices[0]:
                    if idx != -1:  # -1은 유효하지 않은 인덱스
                        # 인덱스를 MIDI ID로 변환 (순서대로 저장되었다고 가정)
                        midi_id = list(midi_metadata.keys())[idx]
                        similar_midi_ids.append(midi_id)
                
                # LLM을 통한 생성 지침 분석
                if input_data.description:
                    prompt = f"""
                    화성 진행과 다음 설명을 기반으로 MIDI 생성을 위한 스타일 지침을 제공해주세요:
                    
                    설명: {input_data.description}
                    
                    JSON 형식으로 다음 정보를 포함하세요:
                    1. recommended_style: 추천 음악 스타일
                    2. melody_characteristics: 멜로디 특성 (예: 점프가 많은, 순차적인 등)
                    3. rhythm_pattern: 리듬 패턴 제안
                    4. additional_notes: 추가 참고사항
                    """
                    
                    llm_response = await query_llm(prompt)
                    try:
                        style_guidance = json.loads(llm_response)
                    except:
                        style_guidance = {"additional_notes": llm_response}
                else:
                    style_guidance = {}
                
                # Magenta로 MIDI 생성
                generated_midis = generate_midi_with_magenta(
                    chord_sequence=midi_bytes,
                    similar_midis=similar_midi_ids,
                    description=input_data.description,
                    temperature=input_data.temperature
                )
                
                if generated_midis:
                    return MIDIResponse(
                        generated_midis=generated_midis,
                        message="MIDI 파일이 성공적으로 생성되었습니다."
                    )
                else:
                    raise HTTPException(status_code=500, detail="MIDI 생성에 실패했습니다.")
            else:
                raise HTTPException(status_code=400, detail="데이터베이스에 MIDI 파일이 없습니다.")
        else:
            raise HTTPException(status_code=500, detail="FAISS 인덱스가 초기화되지 않았습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MIDI 생성 중 오류 발생: {str(e)}")

@app.get("/api/midi-list")
async def get_midi_list():
    """저장된 MIDI 파일 목록 조회"""
    try:
        midi_list = []
        for midi_id, metadata in midi_metadata.items():
            midi_list.append({
                "id": midi_id,
                "filename": metadata.get("filename", "Unknown"),
                "description": metadata.get("description", ""),
                "upload_date": metadata.get("upload_date", "")
            })
        
        return {"midi_files": midi_list, "total_count": len(midi_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MIDI 목록 조회 중 오류 발생: {str(e)}")

# 서버 시작 시 모델 초기화
@app.on_event("startup")
async def startup_event():
    initialize_models()

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)