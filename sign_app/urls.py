from django.urls import path
from . import views
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.urls import reverse_lazy

app_name = 'sign_app'

urlpatterns = [
    path('realtime/', views.realtime_view, name='realtime'),
    path('continuous/', views.continuous_view, name='continuous'),
    path('animation/', views.animation_view, name='animation'),
    path('technology/', views.technology_view, name='technology'),
    path('about/', views.about_view, name='about'),
    path('recognize/', views.recognize_gesture, name='recognize'),
    path('get_gesture_info/', views.get_gesture_info, name='gesture-info'),
    path('generate_animation/', views.generate_animation, name='generate-animation'),  # 新增
    path('get_animation_history/', views.get_animation_history, name='get-animation-history'),
    path('clear_history/', views.clear_history, name='clear-history'),
    path('image_recognition/', views.image_recognition_view, name='image_recognition'),
    path('video_recognition/', views.video_recognition_view, name='video_recognition'),
    path('process_image/', views.handle_image_recognition, name='process_image'),
    path('process_video/', views.handle_video_recognition, name='process_video'),
    path('statistics/', views.statistics_view, name='statistics'),
    path('learning/', views.learning_view, name='learning'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('password_reset/', views.password_reset_request, name='password_reset'),
    path('password_reset/done/', views.password_reset_done, name='password_reset_done'),
    path('reset/<uidb64>/<token>/', views.password_reset_confirm, name='password_reset_confirm'),
    path('reset/done/', views.password_reset_complete, name='password_reset_complete'),
]