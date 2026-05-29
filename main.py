import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torchaudio
import decord
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEModel

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, save_path="./models/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"Validation loss 개선, 모델 저장 :  {self.save_path}")

def create_binary_labels(total_duration_sec, lecture_timestamps):
    labels = np.zeros(total_duration_sec, dtype=np.float32)
    for start, end in lecture_timestamps:
        start_idx = max(0, int(start))
        end_idx = min(total_duration_sec, int(end))
        labels[start_idx:end_idx] = 1.0
    return labels



decord.bridge.set_bridge('torch')

def create_binary_labels(total_duration_sec, lecture_timestamps):
    labels = np.zeros(total_duration_sec, dtype=np.float32)
    for start, end in lecture_timestamps:
        start_idx = max(0, int(start))
        end_idx = min(total_duration_sec, int(end))
        labels[start_idx:end_idx] = 1.0
    return labels


class RealLectureDataset(Dataset):
    def __init__(self, video_data_list, window_size=30, stride=1, cache_dir="./data"):
        """
        Args:
            video_data_list (list): 영상 정보가 담긴 딕셔너리 리스트
            window_size (int): 슬라이딩 윈도우 크기 (second)
            stride (int): 윈도우 이동 간격 (second)
            cache_dir (str): 임베딩 텐서를 저장할 로컬 디렉토리 경로
        """
        self.window_size = window_size
        self.stride = stride
        self.X_samples = []
        self.y_samples = []
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
        print(f"--> Frozen Encoder intiailizing (사용 디바이스: {self.device})")
        
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device)
        self.video_model.eval()
        
        self.audio_model = torch.hub.load('harritaylor/torchvggish', 'vggish').to(self.device)
        self.audio_model.eval()
        
        for data in video_data_list:
            video_path = data["video_path"]
            user_duration = data["duration"]
            timestamps = data["timestamps"]
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_path = os.path.join(cache_dir, f"{video_name}.pt")
            
            labels = create_binary_labels(user_duration, timestamps)
            
            if os.path.exists(cache_path):
                print(f"[{video_name}] Using pre-embedded data: {cache_path}")
                features = torch.load(cache_path, map_location='cpu')
            else:
                print(f"[{video_name}] No pre-embeddded data Adding data. (시간이 다소 소요됨)")
                features = self._extract_features_from_raw_video(video_path, user_duration)
                
                torch.save(features, cache_path)
                print(f"[{video_name}] embedding finished -> {cache_path}")
            
            actual_duration = min(len(features), user_duration)
            
            num_windows = (actual_duration - window_size) // stride + 1
            for i in range(0, num_windows, stride):
                start_t = i
                end_t = i + window_size
                
                window_feat = features[start_t:end_t] # Shape: [30, 896]
                window_label = 1.0 if np.mean(labels[start_t:end_t]) >= 0.5 else 0.0
                
                self.X_samples.append(window_feat)
                self.y_samples.append(window_label)
                
        print(f"=== 총 {len(self.X_samples)}개의 {window_size}초 데이터셋 빌드 완료 ===\n")

    def _extract_features_from_raw_video(self, video_path, duration):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        waveform, sample_rate = torchaudio.load(video_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        combined_features = []
        
        for t in range(duration):
            start_frame = int(t * fps)
            end_frame = int((t + 1) * fps)
            
            if end_frame > total_frames:
                break
                
            frame_indices = np.linspace(start_frame, end_frame - 1, 16, dtype=int)
            video_data = vr.get_batch(frame_indices).permute(0, 3, 1, 2) 
            
            video_inputs = self.video_processor(list(video_data), return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                video_outputs = self.video_model(**video_inputs)
                video_feat = video_outputs.last_hidden_state.mean(dim=1).squeeze(0)
            
            start_sample = int(t * 16000)
            end_sample = int((t + 1) * 16000)
            audio_chunk = waveform[:, start_sample:end_sample]
            
            if audio_chunk.shape[1] < 16000:
                padding = 16000 - audio_chunk.shape[1]
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, padding))
                
            with torch.no_grad():
                audio_feat = self.audio_model(audio_chunk.squeeze(0).to(self.device), 16000)
                if audio_feat.ndim > 1:
                    audio_feat = audio_feat.mean(dim=0) # (128,)
            
            fused_feat = torch.cat([video_feat.cpu(), audio_feat.cpu()], dim=-1) # (896,)
            combined_features.append(fused_feat)
            
        return torch.stack(combined_features)

    def __len__(self):
        return len(self.X_samples)

    def __getitem__(self, idx):
        return self.X_samples[idx], torch.tensor(self.y_samples[idx], dtype=torch.float32)




class RealLectureDataset(Dataset):
    def __init__(self, video_data_list, window_size=30, stride=1):
        """
        Args:
            video_data_list (list): 데이터 리스트
            window_size (int): 슬라이딩 윈도우 크기 (second)
            stride (int): 윈도우 이동 간격 (second)
        """
        self.window_size = window_size
        self.stride = stride
        self.X_samples = []
        self.y_samples = []
        
        # audio & video embeding 후 concat
        # 현재는 (VGGish 128 + VideoMAE 768 = 896)
        # 이후 MLP Projection 넣을수도 있음
        fused_dim = 896 

        for data in video_data_list:
            path = data["video_path"]
            duration = data["duration"]
            timestamps = data["timestamps"]
            
            labels = create_binary_labels(duration, timestamps)
            
            #
            #
            #
            #
            # 이 부분 수정
            #
            #
            #
            features = torch.randn(duration, fused_dim) 
            
            num_windows = (duration - window_size) // stride + 1
            for i in range(0, num_windows, stride):
                start_t = i
                end_t = i + window_size
    
                window_feat = features[start_t:end_t]
                
                window_label = 1.0 if np.mean(labels[start_t:end_t]) >= 0.5 else 0.0
                
                self.X_samples.append(window_feat)
                self.y_samples.append(window_label)

    def __len__(self):
        return len(self.X_samples)

    def __getitem__(self, idx):
        return self.X_samples[idx], torch.tensor(self.y_samples[idx], dtype=torch.float32)


class Lecture1DCNNClassifier(nn.Module):
    def __init__(self, input_dim=896, seq_len=30):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2),  # 30 -> 15
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2)   # 15 -> 7
        )
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (Batch, Dim, Seq)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_loss = val_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return total_loss, acc, precision, recall, f1


if __name__ == "__main__":
    #
    #
    #
    # 나중에 여기다가 직접 저장
    #
    #
    
    my_manually_labeled_data = [
        {
            "video_path": "/home/user/videos/lecture_01.mp4",
            "duration": 600,  # 초단위, 영상 전체 길이
            "timestamps": [(0, 150), (240, 480)]  # 실제 수업이 진행된 구간 (시작초, 종료초) 쌍들
        },
        {
            "video_path": "/home/user/videos/lecture_02.mp4",
            "duration": 900,  # 위와 동일
            "timestamps": [(60, 300), (420, 800)]
        },
        {
            "video_path": "/home/user/videos/lecture_03.mp4",
            "duration": 500,  # validation 용으로 쓸 영상 데이터
            "timestamps": [(0, 200), (300, 450)]
        }
    ]
    
    train_meta = my_manually_labeled_data[:2]
    val_meta = my_manually_labeled_data[2:]
    
    print("dataset build")
    train_dataset = RealLectureDataset(train_meta, window_size=30, stride=1)
    val_dataset = RealLectureDataset(val_meta, window_size=30, stride=1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"학습용 데이터 수: {len(train_dataset)}, 검증용 데이터 수: {len(val_dataset)}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lecture1DCNNClassifier(input_dim=896, seq_len=30).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    early_stopping = EarlyStopping(patience=15, save_path="best_lecture_model.pth")
    
    num_epochs = 100
    print("===== Training ====")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        epoch_val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1-Score: {val_f1:.4f}")
        
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"--> {epoch} Epoch에서 Early Stopping")
            break
            
    print("\n=== 학습 완료 ===")