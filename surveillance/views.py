from django.shortcuts import render, get_object_or_404
from django.http.response import StreamingHttpResponse
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect, FileResponse, Http404, JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .forms import *
from .models import Profile
from django.utils import timezone
from datetime import timedelta
from django.views.decorators.csrf import csrf_exempt
import logging



logger = logging.getLogger(__name__)

from .camera import gen_frames  # Import the frame generator function
from .camera3 import gen_frames3

#Basic list of valid court_id numbers
court_ids = [00000, 00001, 00002, 00003]    #Meant to be 3rd party integrated API to court records, but a simpler example for testing will suffice


# Create your views here.


def index(request):
    return render(request, 'index.html')

@login_required
def base(request):
    logs = Logs.objects.filter(
        Date__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-date')  # Newest first
    
    # Count of new notifications (unread)
    count = logs.filter(is_read=False).count()
    
    context = {
        'logs': logs,
        'count': count,  # This makes {{ count }} available
        'section': 'Notifications',
    }
    return render(request, 'base.html', {'context': context})    
     

#ACCOUNT VIEWS

def register(request):
    user_form = UserRegistrationForm(request.POST)
    if request.method == 'POST':
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            
            # Set the chosen password
            new_user.set_password(user_form.cleaned_data['password'])
            # Save the User object
            new_user.save()
            # Create the user profile
            Profile.objects.create(user=new_user)
            return render(request, 'surveillance/register_done.html', {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'surveillance/register.html', {'user_form': user_form})


def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request, username=cd['username'], password=cd['password'])
        if user is not None:
            if user.is_active:
                login(request, user)
                return render(request, 'surveillance/dashboard.html')
            else:
                return HttpResponse('Disabled account')
        else:
            return HttpResponse('Invalid login')
    else:
        form = LoginForm()
    return render(request, 'surveillance/login.html', {'form': form})


@login_required
def edit(request):

    if request.method == 'POST':
        user_form = UserEditForm(instance=request.user, data=request.POST)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Profile updated successfully')
        else:
            messages.error(request, 'Error updating your profile')
    else:
        user_form = UserEditForm(instance=request.user)
    return render(request, 'surveillance/edit.html', {'user_form': user_form, 'section': 'edit'})



#SECURITY VIEWS


@login_required
def dashboard(request):
    return render(request, 'surveillance/dashboard.html')

@login_required
def cameras(request):
    return render(request, 'surveillance/cameras.html')

@login_required
def camera(request):
    return render(request, 'surveillance/camera.html')


@login_required
def camera2(request):
    return render(request, 'surveillance/camera2.html')

@login_required
def video_feed(request):
    # Return the video feed as an HTTP response
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@login_required
def video_feed2(request):
    # Return the video feed as an HTTP response
    return StreamingHttpResponse(gen_frames3(), content_type='multipart/x-mixed-replace; boundary=frame')



@login_required
def upload_person(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data.get('name')
            img = form.cleaned_data.get('photo1')
            input_court_id = form.cleaned_data.get('court_id')
            if input_court_id in court_ids: 
                obj = Target.objects.create(name=title, photo1=img, court_id = input_court_id)
                obj.save()
                messages.success(request, 'Details successfully uploaded')
            else:
               messages.error(request, 'Invalid Court ID') 
        else:
            messages.error(request, 'Error uploading details')
    else:
        form = UploadForm(request.POST)

        
    return render(request, 'surveillance/upload.html', {'form': form,})


@login_required
def Notifications(request):
    # Get logs from the last 24 hours (or whatever timeframe you prefer)
    logs = Logs.objects.filter(
        date__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-date')  # Newest first
    
    # Count of new notifications (unread)
    new_notifications_count = logs.filter(is_read=False).count()
    
    context = {
        'logs': logs,
        'new_notifications_count': new_notifications_count,
        'current_time': timezone.now(),
    }
    
    return render(request, 'surveillance/notification.html', context)


@login_required
def logs(request):
    # Get logs from the last 24 hours (or whatever timeframe you prefer)
    logs = Logs.objects.all() # Newest first
    
    # Count of new notifications (unread)
    new_notifications_count = logs.filter(is_read=False).count()
    
    context = {
        'logs': logs,
        'new_notifications_count': new_notifications_count,
        'current_time': timezone.now(),
    }
    
    return render(request, 'surveillance/notification.html', context)


@login_required
def view_image(request):
    if request.method == 'GET' and 'photo_id' in request.GET:
        log_entry = get_object_or_404(Logs, id=request.GET.get('photo_id'))
        
        if log_entry.screenshot:
            # Construct absolute path
            image_path = os.path.join(settings.MEDIA_ROOT, str(log_entry.screenshot))
            
            if os.path.exists(image_path):
                return FileResponse(open(image_path, 'rb'), content_type='image/jpeg')
            raise Http404("Image file not found")
        raise Http404("No image available")
    return HttpResponse("Invalid request", status=400)