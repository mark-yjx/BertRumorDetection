# -*- coding: utf-8 -*-
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http import HttpResponse
from django.shortcuts import render

from recognition import rec
from .models import Announcement
from .models import Case
from .models import Dataset
from .models import Popularization
from .models import UserInfo


# Create your views here.

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            user = UserInfo.objects.get(username=username)
            if user.password == password:
                request.session['username'] = username
                return HttpResponse("<script>alert('登录成功');window.location.href='/web/index'</script>")
            # 用户存在且密码匹配，允许用户登录
            # 可以进行后续操作，如设置登录状态等
            else:
                return HttpResponse("<script>alert('用户名或密码错误，请重新输入');window.location.href='/web'</script>")
        # 用户名或密码错误，登录失败
        # 可以返回错误信息或者重定向到登录页面
        except UserInfo.DoesNotExist:
            return HttpResponse("<script>alert('用户名不存在，请重新输入');window.location.href='/web'</script>")
    else:
        return render(request, 'login.html')


def index(request):
    case = Case.objects.all()[:2]
    content = {
        'case': case
    }
    return render(request, 'index.html', content)


def data_list(request):
    dataset_list = Dataset.objects.all()

    paginator = Paginator(dataset_list, 10)  # 每页显示 10 条数据
    page = request.GET.get('page')

    try:
        datasets = paginator.page(page)
    except PageNotAnInteger:
        datasets = paginator.page(1)
    except EmptyPage:
        datasets = paginator.page(paginator.num_pages)

    context = {
        'datasets': datasets
    }
    return render(request, 'data_list.html', context)


def recognize(request):
    if request.method == 'GET':
        return render(request, 'recognize.html', locals())
    elif request.method == 'POST':
        Search = request.POST.get('Search', '')
        result = rec(Search)
        return render(request, 'recognize.html', locals())


def content_analysis(request):
    datas1 = Dataset.objects.filter(lable=1).count()
    datas2 = Dataset.objects.filter(lable=0).count()
    datas3 = Dataset.objects.filter(lable=2).count()
    print(datas1, datas2, datas3)
    dict_value = [{"name": '谣言', "value": datas1}, {"name": '正常', "value": datas2},
                  {"name": '不确定，请谨慎甄别', "value": datas3}]
    return render(request, 'content_analysis.html', locals())


def upload(request):
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        if not file.size:
            # 文件为空，返回错误信息
            error_message = "上传的文件为空，请选择一个有效的文件上传。"
            return render(request, 'upload.html', {'error_message': error_message})

        # 打开一个文件句柄，以二进制方式写入文件内容
        with open('static/upload/file.txt', 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # 读取上传文件
        try:
            with open('static/upload/file.txt', 'rb') as file:
                content = file.read().decode('utf-8')
        except Exception as e:
            # 处理读取文件时发生的异常
            error_message = f"读取文件时出错: {str(e)}"
            return render(request, 'upload.html', {'error_message': error_message})

        if not content:
            # 文件内容为空
            error_message = "上传的文件没有内容，请上传一个非空的文件。"
            return render(request, 'upload.html', {'error_message': error_message})

        sentences = content.split('。')
        result_list = []
        for text in sentences:
            try:
                # rec用于处理文本的函数
                predict = rec(text)
                result_list.append([text, predict])
            except Exception as e:
                # 处理rec函数调用异常
                error_message = f"处理文本 '{text}' 时出错: {str(e)}"
                result_list.append([text, error_message])

        return render(request, 'upload.html', {'result_list': result_list})
    else:
        # 显示文件上传表单页面
        return render(request, 'upload.html')


def popularization(request):
    popularization_list = Popularization.objects.all()
    paginator = Paginator(popularization_list, 5)  # 每页展示5个数据
    page = request.GET.get('page')  # 获取当前页码参数
    popularization_list = paginator.get_page(page)  # 获取当前页的数据对象

    content = {
        'popularization_list': popularization_list
    }
    return render(request, 'popularization.html', content)


def popularization_detail(request):
    popularization_id = request.GET.get('id')
    detail = Popularization.objects.all().filter(id=popularization_id).first()
    content = {
        'detail': detail
    }
    return render(request, 'popularization_detail.html', content)


def case(request):
    case_list = Case.objects.all()

    paginator = Paginator(case_list, 5)  # 每页展示5个数据
    page = request.GET.get('page')  # 获取当前页码参数
    case_list = paginator.get_page(page)  # 获取当前页的数据对象

    content = {
        'case_list': case_list
    }
    return render(request, 'case.html', content)


def case_detail(request):
    case_id = request.GET.get('id')
    case_detail = Case.objects.all().filter(id=case_id).first()
    content = {
        'detail': case_detail
    }
    return render(request, 'case_detail.html', content)


def announcement(request):
    announcement_list = Announcement.objects.all()
    paginator = Paginator(announcement_list, 5)  # 每页展示5个数据
    page = request.GET.get('page')  # 获取当前页码参数
    announcement_list = paginator.get_page(page)  # 获取当前页的数据对象

    content = {
        'announcement_list': announcement_list
    }
    return render(request, 'announcement.html', content)


def announcement_detail(request):
    announcement_id = request.GET.get('id')
    announcement_detail = Announcement.objects.all().filter(id=announcement_id).first()
    content = {
        'detail': announcement_detail
    }
    return render(request, 'announcement_detail.html', content)


def fetch_ul_content(request):
    url = 'https://www.piyao.org.cn/ld.htm'
    base_url = "https://www.piyao.org.cn/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            ul_content = soup.find('ul', {'id': 'list', 'class': 'ej_list xpage-content xpage-content-list'})

            # Convert all relative links to absolute links within the ul_content
            for a_tag in ul_content.find_all('a', href=True):
                a_tag['href'] = urljoin(base_url, a_tag['href'])

            # Similarly, you might want to convert the src of <img> tags or other resources
            for img_tag in ul_content.find_all('img', src=True):
                img_tag['src'] = urljoin(base_url, img_tag['src'])

            return HttpResponse(str(ul_content))
        else:
            return HttpResponse(f"Error fetching content: HTTP Status Code {response.status_code}",
                                status=response.status_code)
    except Exception as e:
        return HttpResponse(f"Error fetching content: {e}", status=500)
