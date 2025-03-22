import cv2
from  metrikz import mse,ssim, snr 
import utility
import pylab
import os
import matplotlib.pyplot as plt

def mpeg2Compress(video_filename, q):
    base = os.path.splitext(video_filename)[0]
    output_file = f'{base}_outputQ{q}.mp4'
    command = [
        'ffmpeg',
        '-y',
        '-an',
        '-i', video_filename,
        '-q:v', str(q),
        '-vcodec', 'mpeg2video',
        output_file
    ]
    utility.execute_command(command)
    return output_file


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def extract_frames(video_input_file, video_output_file):
    """Extrae los frames de los videos original y codificado usando ffmpeg."""
    video_name = os.path.splitext(os.path.basename(video_input_file))[0]
    original_folder = f'./frames/{video_name}/original'
    encoded_folder = f'./frames/{video_name}/encoded'
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(encoded_folder, exist_ok=True)
    
    # Extraer frames del video original
    command = [
        'ffmpeg',
        '-y',
        '-i', video_input_file,
        f'{original_folder}/frame%d.png'
    ]
    utility.execute_command(command)
    
    # Extraer frames del video codificado
    command = [
        'ffmpeg',
        '-y',
        '-i', video_output_file,
        f'{encoded_folder}/frame%d.png'
    ]
    utility.execute_command(command)


def process_video_metrics(video_files, q):
    all_metrics = {}
    
    for video in video_files:
        compressed_file = mpeg2Compress(video, q)
        
        # Extraer los frames de ambos videos
        extract_frames(video, compressed_file)
        
        video_name = os.path.splitext(os.path.basename(video))[0]
        original_folder = f'./frames/{video_name}/original'
        encoded_folder = f'./frames/{video_name}/encoded'
        
        metrics = {'mse': [], 'ssim': [], 'snr': []}
        frame_indices = []
        frame_idx = 1
        
        while True:
            original_frame_path = f'{original_folder}/frame{frame_idx}.png'
            encoded_frame_path = f'{encoded_folder}/frame{frame_idx}.png'
            
            if not os.path.exists(original_frame_path) or not os.path.exists(encoded_frame_path):
                break
            
            # Leer los frames extraídos con OpenCV
            frame_orig = cv2.imread(original_frame_path, cv2.IMREAD_GRAYSCALE)
            frame_comp = cv2.imread(encoded_frame_path, cv2.IMREAD_GRAYSCALE)
            
            # Calcular métricas
            metrics['mse'].append(mse(frame_orig, frame_comp))
            metrics['ssim'].append(ssim(frame_orig, frame_comp))
            metrics['snr'].append(snr(frame_orig, frame_comp))
            frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        all_metrics[video] = {'metrics': metrics, 'frames': frame_indices}
    
    # Graficas: MSE, SSIM y SNR vs. frame
    colors = ['r', 'g', 'b', 'c', 'm']
    plt.figure(figsize=(15, 5))
    
    # MSE
    plt.subplot(1, 3, 1)
    for i, video in enumerate(video_files):
        frames = all_metrics[video]['frames']
        mse_vals = all_metrics[video]['metrics']['mse']
        plt.plot(frames, mse_vals, color=colors[i % len(colors)], label=os.path.basename(video))
    plt.title("MSE vs. Frame")
    plt.xlabel("Frame")
    plt.ylabel("MSE")
    plt.legend()
    
    # SSIM
    plt.subplot(1, 3, 2)
    for i, video in enumerate(video_files):
        frames = all_metrics[video]['frames']
        ssim_vals = all_metrics[video]['metrics']['ssim']
        plt.plot(frames, ssim_vals, color=colors[i % len(colors)], label=os.path.basename(video))
    plt.title("SSIM vs. Frame")
    plt.xlabel("Frame")
    plt.ylabel("SSIM")
    plt.legend()
    
    # SNR
    plt.subplot(1, 3, 3)
    for i, video in enumerate(video_files):
        frames = all_metrics[video]['frames']
        snr_vals = all_metrics[video]['metrics']['snr']
        plt.plot(frames, snr_vals, color=colors[i % len(colors)], label=os.path.basename(video))
    plt.title("SNR vs. Frame")
    plt.xlabel("Frame")
    plt.ylabel("SNR (dB)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()