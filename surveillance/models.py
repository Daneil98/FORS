from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
import time, os
import cv2
from django.core.files import File
from twilio.rest import Client


# Create your models here.

account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)


#USER ACCOUNT MODEL
class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    
    def __str__(self):
        return f'Profile for user {self.user.username}'
    
    
class Target(models.Model):
    name = models.CharField(max_length=30, null=False)
    photo1 = models.ImageField(upload_to='', blank=False)
    photo2 = models.ImageField(upload_to='', blank=True)
    
class Logs(models.Model):
    person = models.CharField(max_length=10, null=False)
    weapon = models.CharField(max_length=10, null=True)
    camera = models.IntegerField(null=True)
    date =  models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    screenshot = models.ImageField(upload_to='screenshots/', blank=True, null=True)
    
    @classmethod
    def capture_screenshot(cls, frame, detection_type):
        """Capture and save screenshot from OpenCV frame"""
        try:
            # Create screenshots directory if it doesn't exist
            os.makedirs('media/screenshots', exist_ok=True)
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{detection_type}_{timestamp}.jpg"
            full_path = os.path.join(settings.MEDIA_ROOT, 'screenshots', filename)
            
            # Save the image
            cv2.imwrite(full_path, frame)
            
            # Return both path and filename
            return full_path, filename
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None, None

    @classmethod
    def create_if_not_recent(cls, person, camera, weapon='', frame=None):
        """Create log entry if no recent detection exists"""
        recent_log = cls.objects.filter(
            person=person,
            date__gte=timezone.now() - timedelta(minutes=30),
            camera=camera
        ).exists()
        
        if not recent_log:
            log = cls(person=person, weapon=weapon, camera=camera)
            
            if frame is not None:
                # Capture screenshot
                screenshot_path, filename = cls.capture_screenshot(frame, f"person_{person}")
                
                if screenshot_path:
                    # Just assign the relative path
                    log.screenshot = screenshot_path
                
            log.save()

            message = client.messages.create(
                body = f"{person} or {weapon} was detected in camera {camera}",
                from_='',
                to=''
            )

            print(message.sid)
            return log
        return None
