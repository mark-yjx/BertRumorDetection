from django.db import models


# Create your models here.


class UserInfo(models.Model):
    id = models.IntegerField(verbose_name='编号', primary_key=True)
    username = models.CharField(verbose_name='用户名', max_length=255)
    password = models.CharField(verbose_name='密码', max_length=255)


class Case(models.Model):
    id = models.IntegerField(verbose_name='编号', primary_key=True)
    title = models.CharField(verbose_name='标题', max_length=255)
    section = models.CharField(verbose_name='部分内容', max_length=255)
    content = models.TextField(verbose_name='内容')
    img_url = models.CharField(verbose_name='图片链接', max_length=255)


class Dataset(models.Model):
    id = models.IntegerField(verbose_name='主键', primary_key=True)
    text = models.TextField(verbose_name='短信内容', default='')
    lable = models.IntegerField(verbose_name='分类标签', default=0)
    type = models.CharField(verbose_name='已读类型', max_length=255)
    date = models.CharField(verbose_name='时间', max_length=255, default='暂无')


class Popularization(models.Model):
    id = models.IntegerField(verbose_name='主键', primary_key=True)
    title = models.CharField(verbose_name='标题', max_length=255)
    img_url = models.CharField(verbose_name='图片地址', max_length=255)
    content = models.TextField(verbose_name='知识科普内容')

class Announcement(models.Model):
    id = models.IntegerField(verbose_name='编号', primary_key=True)
    title = models.CharField(verbose_name='标题', max_length=255)
    img_url = models.CharField(verbose_name='图片地址', max_length=255)
    content = models.TextField(verbose_name='公告内容', max_length=255)
