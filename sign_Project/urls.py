# """
# URL configuration for sign_Project project.
#
# The `urlpatterns` list routes URLs to views. For more information please see:
#     https://docs.djangoproject.com/en/5.2/topics/http/urls/
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to urlpatterns:  path('', views.home, name='home')
# Class-based views
#     1. Add an import:  from other_app.views import Home
#     2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
# Including another URLconf
#     1. Import the include() function: from django.urls import include, path
#     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
# """
#
# from django.contrib import admin
# from django.urls import path,include
# from django.views.generic import RedirectView
# from django.conf import settings
# from django.conf.urls.static import static
# from sign_app import views as sign_views
#
# urlpatterns = [
#     path("admin/", admin.site.urls),
#     path('', include('sign_app.urls')),
#
#     path('password_reset/', sign_views.password_reset_request, name='password_reset'),
#               ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT),
#
#
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

"""
URL configuration for sign_Project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from sign_app import views as sign_views  # 保留密码重置视图

urlpatterns = [
    # 根路径重定向到登录页 (来自第二个文件)
    path('', RedirectView.as_view(url='/login/')),

    # 应用路由配置 (来自第一个文件)
    path('', include('sign_app.urls')),

    # 管理后台 (两个文件共有)
    path("admin/", admin.site.urls),

    # 密码重置功能 (来自第一个文件)
    path('password_reset/', sign_views.password_reset_request, name='password_reset'),
]

# 静态文件服务 (来自第一个文件)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# 开发环境下的媒体文件服务 (结合两个文件的处理方式)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)