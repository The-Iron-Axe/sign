from django.db import models
# 注册新加

from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _


class CustomUser(AbstractUser):
    email = models.EmailField(_('email address'), unique=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # 添加任何其他自定义字段
    def __str__(self):
        return self.username
#注册新加
class SignHistory(models.Model):
    text = models.CharField(max_length=255)
    video_file = models.CharField(max_length=255)
    lang = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.text} ({self.lang}) - {self.created_at}"

class ActivityLog(models.Model):
    """用于记录系统使用活动日志的模型"""
    ACTIVITY_TYPES = [
        ('REALTIME_RECOGNITION', '孤立手语识别'),
        ('IMAGE_RECOGNITION', '中文图片识别'),
        ('VIDEO_RECOGNITION', '中文视频识别'),
        ('ANIMATION_GENERATION', '手语动画生成'),
    ]

    activity_type = models.CharField(max_length=50, choices=ACTIVITY_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.JSONField(default=dict)  # 使用JSONField存储额外信息，如识别结果、置信度等
    lang = models.CharField(max_length=10, null=True, blank=True) # 记录语言

    def __str__(self):
        return f"{self.get_activity_type_display()} at {self.timestamp}"