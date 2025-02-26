from django.db import models
from django.conf import settings

# Create your models here.


#USER ACCOUNT MODEL
class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    
    def __str__(self):
        return f'Profile for user {self.user.username}'
    
    
class Target(models.Model):
    name = models.CharField(max_length=10, null=False)
    photo1 = models.ImageField(upload_to='media/', null=True, blank=False)
    photo2 = models.ImageField(upload_to='media/', null=True, blank=True)
    photo3 = models.ImageField(upload_to='media/', null=True, blank=True)