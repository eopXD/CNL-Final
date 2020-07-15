from AIWeb.utils import *
from django.db import models
from django.utils import timezone
from django.forms.models import model_to_dict

class Table1(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

class BaseModel(models.Model):
    class Meta:
        abstract = True
    def to_dict(self):
        return model_to_dict(self)
    def __str__(self):
        return json.dumps(model_to_dict(self), indent=1)
    def __repr__(self):
        return json.dumps(model_to_dict(self))
# ================================================================
class RawData(BaseModel):
    created = models.DateTimeField(auto_now_add=True)
    pi = models.TextField('source PI', default='unknown')
    target = models.TextField('csv content', default='')

class Parsed(BaseModel):
    raw_id = models.IntegerField(default=-1)
    created = models.DateTimeField(auto_now_add=True)
    parser = models.TextField('parser tag', default='noparser')
    pi = models.TextField('source PI', default='unknown')
    mac = models.TextField('source MAC address', default='unknown')
    target = models.TextField('parsed csv content', default='')

class PosQuery(BaseModel):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    mac = models.TextField('source MAC address', default='unknown')

class PosPredict(BaseModel):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    mac = models.TextField('source MAC address', default='unknown')
    tag = models.TextField(default='')
    x = models.FloatField(default=-1)
    y = models.FloatField(default=-1)





