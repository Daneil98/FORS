from django.urls import path
from surveillance import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.index, name='index'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logged_out/', auth_views.LogoutView.as_view(), name='logged_out'),
    path('password_change/', auth_views.PasswordChangeView.as_view(), name='password_change'),
    path('password_change/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('register/', views.register, name='register'),
    path('edit/', views.edit, name='edit'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('video_feed2/', views.video_feed2, name='video_feed2'),
    path('cameras/', views.cameras, name='cameras'),
    path('camera/', views.camera, name='camera'),
    path('camera2/', views.camera2, name='camera2'),
    path('upload/', views.upload_person, name='upload'),
    path('Notifications/', views.Notifications, name='Notifications'),
    path('Logs/', views.logs, name='Logs'),
    path('view-image/', views.view_image, name='view_image'),
]