from django.core.management.base import BaseCommand
from sign_app.models import SignHistory
import os, shutil
from django.conf import settings

class Command(BaseCommand):
    help = '清空历史记录并删除output目录下所有视频文件'

    def handle(self, *args, **kwargs):
        # 清空历史记录
        SignHistory.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('已清空SignHistory表'))
        # 删除output目录下所有文件
        output_dir = os.path.join(settings.BASE_DIR, 'static', 'output')
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                file_path = os.path.join(output_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'删除{file_path}失败: {e}'))
            self.stdout.write(self.style.SUCCESS('已删除output目录下所有视频文件'))
        else:
            self.stdout.write('output目录不存在') 