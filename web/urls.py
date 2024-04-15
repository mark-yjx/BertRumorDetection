from django.urls import path

from . import views

app_name = 'web'

urlpatterns = [
    # 登录
    path(r'', views.login, name='login'),
    path('login/', views.login, name='login'),
    # 首页
    path(r'index/', views.index, name='index'),
    # 数据查看
    path(r'data_list/', views.data_list, name='data_list'),
    # 短信识别
    path(r'recognize/', views.recognize, name='recognize'),
    # 短信内容分析
    path(r'content_analysis/', views.content_analysis, name='content_analysis'),
    # 上传文件识别
    path(r'upload/', views.upload, name='upload'),
    # 谣言辨别知识科普
    path(r'popularization/', views.popularization, name='popularization'),
    # 谣言辨别知识详情
    path(r'popularization_detail/', views.popularization_detail, name='popularization_detail'),
    # 短信诈骗案例
    path(r'case/', views.case, name='case'),
    # 短信诈骗详情
    path(r'case_detail/', views.case_detail, name='case_detail'),
    # 公告
    path(r'announcement/', views.announcement, name='announcement'),
    # 公告详情
    path(r'announcement_detail/', views.announcement_detail, name='announcement_detail'),
    path('fetch-ul/', views.fetch_ul_content, name='fetch_ul'),
]
