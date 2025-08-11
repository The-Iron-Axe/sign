from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
import os
import json
from django.shortcuts import render
from .gesture_model import GestureRecognizer
from sign_Project.settings import BASE_DIR
import jieba
import re
from datetime import datetime
from django.conf import settings
import tempfile
from PIL import Image
import subprocess
from .models import SignHistory, ActivityLog
from django.views.decorators.http import require_GET
import torch
from torchvision import transforms
import mediapipe as mp
from .CNN_LSTM import FeatureExtractor, MultiModalCNNTransformerModel
from typing import Optional, List
import random
import difflib  # 恢复 difflib 导入，为英文逻辑服务
from django.db.models import Count
from django.db.models.functions import TruncDate

# --- 修改点：全局记忆变量现在只为中文演示模式服务 ---
word_to_image_memory = {}
# --- 修改结束 ---

# 连续手语识别模型全局变量 (保持不变)
continuous_model = None
continuous_feature_extractor = None
continuous_label_map = None
continuous_mp_hands = None


def load_continuous_model():
    """加载连续手语识别模型 (保持不变)"""
    global continuous_model, continuous_feature_extractor, continuous_label_map, continuous_mp_hands
    if continuous_model is None:
        try:
            model_path = os.path.join(BASE_DIR, 'ctcn_2', 'models', 'best_model.pth')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            continuous_feature_extractor = FeatureExtractor(model_name='resnet18')
            continuous_feature_extractor.eval()
            continuous_model = MultiModalCNNTransformerModel(
                feature_dim=638,
                num_classes=len(checkpoint['index_to_label']),
                heads=2
            )
            continuous_model.load_state_dict(checkpoint['model_state_dict'])
            continuous_model.eval()
            continuous_label_map = {int(k): v for k, v in checkpoint['index_to_label'].items()}
            continuous_mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2
            )
            print("连续手语识别模型加载成功")
        except Exception as e:
            print(f"加载连续手语识别模型失败: {str(e)}")


load_continuous_model()


# ... (preprocess_frame, extract_keypoints 函数保持不变) ...
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return transform(pil_image)


def extract_keypoints(frame, mp_hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = mp_hands.process(frame_rgb)
    keypoints = np.zeros(126, dtype=np.float32)
    if hands_results.multi_hand_landmarks:
        hands = hands_results.multi_hand_landmarks[:2]
        for hand_idx, hand in enumerate(hands):
            start = hand_idx * 63
            for lm_idx, landmark in enumerate(hand.landmark[:21]):
                pos = start + lm_idx * 3
                if pos + 2 < 126:
                    keypoints[pos] = landmark.x
                    keypoints[pos + 1] = landmark.y
                    keypoints[pos + 2] = landmark.z
    return torch.tensor(keypoints, dtype=torch.float32)


# --- 英文处理逻辑 (恢复) ---

def find_similar_word(word, word_list, cutoff=0.8):
    """查找相似词 (为英文逻辑服务)"""
    matches = difflib.get_close_matches(word, word_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def get_sign_file_en(word, word_list, base_dir=None):
    """英文：优先查找mp4，没有则查png，再没有查每个字母的png"""
    if base_dir is None:
        base_dir = os.path.join(settings.BASE_DIR, 'static', 'database', 'words')
    if word in [',', '.', '!', '?', ';', ' ']:
        return None
    # 优先查找单词.mp4
    mp4_file = os.path.join(base_dir, f'{word}.mp4')
    if os.path.exists(mp4_file):
        return mp4_file
    # 查找相似词.mp4
    similar_word = find_similar_word(word, word_list)
    if similar_word:
        mp4_file = os.path.join(base_dir, f'{similar_word}.mp4')
        if os.path.exists(mp4_file):
            return mp4_file
    # 查找单词.png
    png_file = os.path.join(base_dir, f'{word}.png')
    if os.path.exists(png_file):
        return [png_file]
    # 查找相似词.png
    if similar_word:
        png_file = os.path.join(base_dir, f'{similar_word}.png')
        if os.path.exists(png_file):
            return [png_file]
    # 查找每个字母.png
    letter_dir = os.path.join(settings.BASE_DIR, 'static', 'database', 'alphabet')
    letter_files = []
    for letter in word:
        letter_file = os.path.join(letter_dir, f'{letter.lower()}.png')
        if os.path.exists(letter_file):
            letter_files.append(letter_file)
    return letter_files if letter_files else None


def create_sign_language_sequence_en(words, word_list):
    """英文：支持mp4和png混合合成，输出为mp4格式"""
    output_dir = os.path.join(settings.BASE_DIR, 'static', 'output')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    temp_output_filename = f"temp_sign_language_{timestamp}.mp4"
    temp_video_path = os.path.join(output_dir, temp_output_filename)

    width, height = 640, 480
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    final_output_filename = f"sign_language_{timestamp}_h264.mp4"
    final_video_path = os.path.join(output_dir, final_output_filename)

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        for word in words:
            sign_file = get_sign_file_en(word, word_list)
            if isinstance(sign_file, str) and sign_file.endswith('.mp4'):
                cap = cv2.VideoCapture(sign_file)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(cv2.resize(frame, (width, height)))
                cap.release()
            elif isinstance(sign_file, list) and sign_file:
                for image_path in sign_file:
                    img = cv2.imread(image_path)
                    if img is None:
                        for _ in range(fps): out.write(black_frame)
                        continue
                    resized_img = cv2.resize(img, (width, height))
                    for _ in range(fps): out.write(resized_img)
            else:
                for _ in range(fps): out.write(black_frame)
    except Exception as e:
        print(f"生成英文视频出错: {e}")
        out.release()
        return None
    finally:
        out.release()

    try:
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            return None
        convert_mp4_to_h264(temp_video_path, final_video_path)
        return final_video_path
    except Exception as e:
        print(f"FFmpeg 转码失败: {e}")
        return None


# --- 中文演示逻辑 (保持不变) ---
def tokenize_text(text, lang='en'):
    """根据语言分词"""
    if lang == 'zh':
        tokens = []
        for fragment in re.split(r'([，。！？；])', text):
            if fragment and fragment in '，。！？；':
                tokens.append(fragment)
            elif fragment:
                tokens.extend(jieba.lcut(fragment))
        return [token for token in tokens if token.strip()]
    else:
        return re.findall(r"[\w']+|[.,!?;]", text.lower())


def get_image_path_for_word_demo(word, available_images):
    """为中文演示模式获取图片路径"""
    global word_to_image_memory
    if word in '，。！？；,.;?!':
        return None
    if word in word_to_image_memory:
        return word_to_image_memory[word]

    words_dir_zh = os.path.join(settings.BASE_DIR, 'static', 'database', 'words-zhcn')
    image_path = os.path.join(words_dir_zh, f'{word}.png')
    if os.path.exists(image_path):
        word_to_image_memory[word] = image_path
        if image_path in available_images:
            available_images.remove(image_path)
        return image_path

    if available_images:
        random_image_path = random.choice(available_images)
        available_images.remove(random_image_path)
        word_to_image_memory[word] = random_image_path
        return random_image_path

    return None


def create_sign_language_video_demo(words):
    """为中文演示模式创建视频"""
    words_dir_zh = os.path.join(settings.BASE_DIR, 'static', 'database', 'words-zhcn')
    all_images = [os.path.join(words_dir_zh, f) for f in os.listdir(words_dir_zh) if f.endswith('.png')]
    random.shuffle(all_images)
    used_by_memory = set(word_to_image_memory.values())
    available_random_pool = [img for img in all_images if img not in used_by_memory]

    image_paths_for_video = [get_image_path_for_word_demo(word, available_random_pool) for word in words if
                             get_image_path_for_word_demo(word, available_random_pool)]

    if not image_paths_for_video:
        return None

    output_dir = os.path.join(settings.BASE_DIR, 'static', 'output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_video_path = os.path.join(output_dir, f"temp_sign_language_zh_{timestamp}.mp4")

    width, height, fps, duration = 640, 480, 30, 1.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    for image_path in image_paths_for_video:
        try:
            img = cv2.imread(image_path)
            if img is None: raise IOError
            resized_img = cv2.resize(img, (width, height))
            for _ in range(int(fps * duration)): out.write(resized_img)
            for _ in range(int(fps * 0.2)): out.write(black_frame)
        except Exception as e:
            print(f"处理图片 '{image_path}' 时出错: {e}")
            for _ in range(int(fps * duration)): out.write(black_frame)

    out.release()

    final_video_path = temp_video_path.replace('.mp4', '_h264.mp4')
    convert_mp4_to_h264(temp_video_path, final_video_path)
    return final_video_path


def convert_mp4_to_h264(input_path, output_path):
    cmd = ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-an', output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(input_path): os.remove(input_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg转码失败: {e}")
        if os.path.exists(output_path): os.remove(output_path)
        os.rename(input_path, output_path)


# --- 核心视图函数 (修改点) ---
@csrf_exempt
def generate_animation(request):
    """根据语言分流处理动画生成请求"""
    if request.method == 'POST':
        try:
            text = request.POST.get('text', '').strip()
            lang = request.POST.get('lang', 'en').strip()
            if not text:
                return JsonResponse({'error': '请输入要翻译的文本'}, status=400)

            output_video = None
            if lang == 'zh':
                # --- 中文走演示逻辑 ---
                words = tokenize_text(text, 'zh')
                output_video = create_sign_language_video_demo(words)
            else:
                # --- 英文走原有逻辑 ---
                words_dir_en = os.path.join(settings.BASE_DIR, 'static', 'database', 'words')
                word_list_en = [os.path.splitext(f)[0] for f in os.listdir(words_dir_en) if
                                f.endswith(('.mp4', '.png'))]
                words = tokenize_text(text, 'en')
                output_video = create_sign_language_sequence_en(words, word_list_en)

            if output_video and os.path.exists(output_video):
                video_filename = os.path.basename(output_video)
                video_url = f"{settings.STATIC_URL}output/{video_filename}"

                # 保存历史和日志
                SignHistory.objects.create(text=text, video_file=video_url, lang=lang)
                ActivityLog.objects.create(
                    activity_type='ANIMATION_GENERATION',
                    lang=lang,
                    details={'text': text, 'video_url': video_url}
                )
                return JsonResponse({'video_url': video_url})
            else:
                return JsonResponse({'error': '无法生成手语动画'}, status=500)
        except Exception as e:
            print(f"生成动画时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': f'服务器内部错误: {e}'}, status=500)

    return JsonResponse({'error': '无效的请求方法'}, status=400)


# ... (其余所有视图函数和代码保持不变) ...

'''下面为手语翻译的代码'''
# 手势类别字典
classes_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,
    'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}

# 加载手势详细信息
gesture_info_path = os.path.join(BASE_DIR, 'model', 'gesture_labels.json')
try:
    with open(gesture_info_path, 'r', encoding='utf-8') as f:
        gesture_info_by_index = json.load(f)
    gesture_info = {info['symbol']: {'name': info['name'], 'description': info['description']}
                    for _, info in gesture_info_by_index.items()}
    gesture_info['None'] = {'name': '未检测到手势', 'description': '请确保手在摄像头范围内'}
except Exception as e:
    print(f"加载手势信息文件时出错: {e}")
    gesture_info = {letter: {'name': f'字母 {letter}', 'description': f'这是字母 {letter} 的手势'}
                    for letter in classes_dict}
    gesture_info['None'] = {'name': '未检测到手势', 'description': '请确保手在摄像头范围内'}

# 初始化手势识别器
model_path = os.path.join(os.path.dirname(__file__), '../model/CNN_model_alphabet_SIBI.pth')
recognizer = GestureRecognizer(model_path, classes_dict)


def realtime_view(request):
    return render(request, 'realtime.html')


@csrf_exempt
def get_gesture_info(request):
    return JsonResponse(gesture_info)


@csrf_exempt
def recognize_gesture(request):
    if request.method == 'POST':
        try:
            data = request.POST.get('image')
            if not data: return JsonResponse({'error': '未提供图像数据'}, status=400)
            if 'base64,' in data: data = data.split('base64,')[1]
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gesture_symbol, confidence, bbox = recognizer.recognize_gesture(image)
            response_data = {}
            if gesture_symbol is None:
                info = gesture_info['None']
                response_data = {'gesture': 'None', 'name': info['name'], 'description': info['description'],
                                 'confidence': 0.0}
            else:
                info = gesture_info.get(gesture_symbol,
                                        {'name': gesture_symbol, 'description': f'手势 {gesture_symbol}'})
                response_data = {'gesture': gesture_symbol, 'name': info['name'], 'description': info['description'],
                                 'confidence': confidence, 'bbox': bbox}
                ActivityLog.objects.create(activity_type='REALTIME_RECOGNITION', lang='en',
                                           details={'gesture': gesture_symbol, 'confidence': confidence})
            return JsonResponse(response_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': '无效的请求方法'}, status=400)


def continuous_view(request): return render(request, 'continuous.html')


def animation_view(request): return render(request, 'animation.html')


def technology_view(request): return render(request, 'technology.html')


def about_view(request): return render(request, 'about.html')


@require_GET
def get_animation_history(request):
    records = SignHistory.objects.order_by('-created_at')[:10]
    data = [{'text': r.text, 'video_file': r.video_file, 'lang': r.lang,
             'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S')} for r in records]
    return JsonResponse({'history': data})


@csrf_exempt
def clear_history(request):
    if request.method == 'POST':
        SignHistory.objects.all().delete()
        global word_to_image_memory
        word_to_image_memory.clear()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Invalid method'})


def image_recognition_view(request): return render(request, 'image_recognition.html')


def video_recognition_view(request): return render(request, 'video_recognition.html')


@csrf_exempt
def handle_video_recognition(request):
    if request.method == 'POST':
        load_continuous_model()
        if 'video' not in request.FILES: return JsonResponse({'error': '未上传视频文件'}, status=400)
        video_file = request.FILES['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in video_file.chunks(): tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        try:
            cap, frames, keypoints_list, frame_count, sample_rate = cv2.VideoCapture(tmp_file_path), [], [], 0, 3
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                if frame_count % sample_rate != 0: continue
                frames.append(preprocess_frame(frame))
                keypoints_list.append(extract_keypoints(frame, continuous_mp_hands))
            cap.release()
            if not frames: return JsonResponse({'error': '未检测到有效视频帧'}, status=400)
            frames_tensor, keypoints_tensor = torch.stack(frames), torch.stack(keypoints_list)
            with torch.no_grad():
                visual_features = continuous_feature_extractor(frames_tensor.view(-1, 3, 256, 256)).view(1, -1, 512)
                combined_features = torch.cat([visual_features, keypoints_tensor.unsqueeze(0)], dim=2)
                seq_len = combined_features.shape[1]
                if seq_len < 170:
                    combined_features = torch.cat([combined_features, torch.zeros(1, 170 - seq_len, 638)], dim=1)
                else:
                    combined_features = combined_features[:, :170]
                outputs = continuous_model(combined_features)
                probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs).item()
            label, confidence = continuous_label_map.get(pred_idx, "未知标签"), probs[0][pred_idx].item()
            ActivityLog.objects.create(activity_type='VIDEO_RECOGNITION', lang='zh',
                                       details={'result': label, 'confidence': confidence})
            return JsonResponse({'result': label, 'confidence': f"{confidence:.2%}"})
        except Exception as e:
            return JsonResponse({'error': f"处理错误: {str(e)}"}, status=500)
        finally:
            if os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
    return JsonResponse({'error': '无效的请求方法'}, status=400)


@csrf_exempt
def handle_image_recognition(request):
    if request.method == 'POST':
        load_continuous_model()
        if 'images' not in request.FILES: return JsonResponse({'error': '未上传图片文件'}, status=400)
        image_files = sorted(request.FILES.getlist('images'), key=lambda x: x.name)
        frames, keypoints_list = [], []
        try:
            for image_file in image_files:
                frame = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
                frames.append(preprocess_frame(frame))
                keypoints_list.append(extract_keypoints(frame, continuous_mp_hands))
            if not frames: return JsonResponse({'error': '未检测到有效图片'}, status=400)
            frames_tensor, keypoints_tensor = torch.stack(frames), torch.stack(keypoints_list)
            with torch.no_grad():
                visual_features = continuous_feature_extractor(frames_tensor.view(-1, 3, 256, 256)).view(1, -1, 512)
                combined_features = torch.cat([visual_features, keypoints_tensor.unsqueeze(0)], dim=2)
                seq_len = combined_features.shape[1]
                if seq_len < 170:
                    combined_features = torch.cat([combined_features, torch.zeros(1, 170 - seq_len, 638)], dim=1)
                else:
                    combined_features = combined_features[:, :170]
                outputs = continuous_model(combined_features)
                probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs).item()
            label, confidence = continuous_label_map.get(pred_idx, "未知标签"), probs[0][pred_idx].item()
            ActivityLog.objects.create(activity_type='IMAGE_RECOGNITION', lang='zh',
                                       details={'result': label, 'confidence': confidence})
            return JsonResponse({'result': label, 'confidence': f"{confidence:.2%}"})
        except Exception as e:
            return JsonResponse({'error': f"处理错误: {str(e)}"}, status=500)
    return JsonResponse({'error': '无效的请求方法'}, status=400)


def statistics_view(request):
    total_activities = ActivityLog.objects.count()
    activities_by_type = ActivityLog.objects.values('activity_type').annotate(count=Count('id')).order_by('-count')
    type_counts = {item['activity_type']: item['count'] for item in activities_by_type}
    daily_counts_query = ActivityLog.objects.annotate(date=TruncDate('timestamp')).values('date').annotate(
        count=Count('id')).order_by('date')
    labels = [item['date'].strftime('%Y-%m-%d') for item in daily_counts_query]
    data = [item['count'] for item in daily_counts_query]
    recent_activities = ActivityLog.objects.order_by('-timestamp')[:10]
    context = {
        'total_activities': total_activities,
        'realtime_count': type_counts.get('REALTIME_RECOGNITION', 0),
        'image_count': type_counts.get('IMAGE_RECOGNITION', 0),
        'video_count': type_counts.get('VIDEO_RECOGNITION', 0),
        'animation_count': type_counts.get('ANIMATION_GENERATION', 0),
        'chart_labels': json.dumps(labels),
        'chart_data': json.dumps(data),
        'recent_activities': recent_activities,
    }
    return render(request, 'statistics.html', context)

def learning_view(request):
    """渲染学习模式页面"""
    return render(request, 'learning.html')

def login_view(request):
    """渲染登录页面"""
    return render(request, 'login.html')


#login_page---

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.urls import reverse
from .forms import RegisterForm, LoginForm


def register_view(request):
    if request.user.is_authenticated:
        return redirect('sign_app:realtime')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '注册成功！欢迎使用双向手语翻译系统')
            return redirect('sign_app:realtime')
    else:
        form = RegisterForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('sign_app:realtime')

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            remember_me = form.cleaned_data.get('remember_me')

            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)

                # 处理"记住我"功能
                if not remember_me:
                    request.session.set_expiry(0)  # 浏览器关闭时过期

                messages.success(request, f'欢迎回来，{username}！')
                return redirect('sign_app:realtime')
        else:
            messages.error(request, '用户名或密码不正确')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})


@login_required
def logout_view(request):
    logout(request)
    messages.info(request, '您已成功登出')
    return redirect('sign_app:login')


@login_required
def realtime_view(request):
    return render(request, 'realtime.html')

from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.urls import reverse_lazy

def password_reset_request(request):
    return PasswordResetView.as_view(
        template_name='password_reset.html',
        email_template_name='password_reset_email.html',
        subject_template_name='password_reset_subject.txt',
        success_url=reverse_lazy('sign_app:password_reset_done')
    )(request)


#密码重置试图
class CustomPasswordResetView(PasswordResetView):
    template_name = 'password_reset.html'
    email_template_name = 'password_reset_email.html'
    subject_template_name = 'password_reset_subject.txt'
    success_url = reverse_lazy('sign_app:password_reset_done')

class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'password_reset_done.html'

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'password_reset_confirm.html'
    success_url = reverse_lazy('sign_app:password_reset_complete')

class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'password_reset_complete.html'

# 包装函数视图
def password_reset_request(request):
    return CustomPasswordResetView.as_view()(request)

def password_reset_done(request):
    return CustomPasswordResetDoneView.as_view()(request)

def password_reset_confirm(request, uidb64, token):
    return CustomPasswordResetConfirmView.as_view()(request, uidb64=uidb64, token=token)

def password_reset_complete(request):
    return CustomPasswordResetCompleteView.as_view()(request)
