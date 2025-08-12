from django.contrib import admin
from .models import *
# Register your models here.


@admin.register(Target)
class TargetAdmin(admin.ModelAdmin):
    list_display = ['name', 'photo1', 'photo2']
    
@admin.register(Logs)
class LogsAdmin(admin.ModelAdmin):
    list_display = ['person', 'weapon', 'camera', 'date']