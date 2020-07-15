# Generated by Django 2.1.3 on 2019-06-11 09:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AIWeb', '0002_rawdata'),
    ]

    operations = [
        migrations.CreateModel(
            name='Parsed',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('parser', models.TextField(default='noparser', verbose_name='parser tag')),
                ('pi', models.TextField(default='unknown', verbose_name='source PI')),
                ('mac', models.TextField(default='unknown', verbose_name='source MAC address')),
                ('target', models.TextField(default='', verbose_name='parsed csv content')),
            ],
        ),
        migrations.CreateModel(
            name='PosQuery',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('mac', models.TextField(default='unknown', verbose_name='source MAC address')),
            ],
        ),
        migrations.AlterField(
            model_name='rawdata',
            name='created',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
