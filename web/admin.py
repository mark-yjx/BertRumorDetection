from django.contrib import admin
from .models import *


# Register your models here.
class UnderwriterDataset(admin.ModelAdmin):
    # 需要显示的字段信息
    list_display = ('id', 'text', 'lable')


class UnderCase(admin.ModelAdmin):
    # 需要显示的字段信息
    list_display = ('id', 'title', 'section', 'content', 'img_url')


class UnderUserInfo(admin.ModelAdmin):
    # 需要显示的字段信息
    list_display = ('id', 'username', 'password')


class UnderPopularization(admin.ModelAdmin):
    # 需要显示的字段信息
    list_display = ('id', 'title', 'img_url', 'content')


class UnderAnnouncement(admin.ModelAdmin):
    # 需要显示的字段信息
    list_display = ('id', 'title', 'img_url', 'content')


admin.site.site_title = "谣言检测系统"
admin.site.site_header = "谣言检测系统"
admin.site.register(Dataset, UnderwriterDataset)
admin.site.register(Case, UnderCase)
admin.site.register(UserInfo, UnderUserInfo)
admin.site.register(Popularization, UnderPopularization)
admin.site.register(Announcement, UnderAnnouncement)
