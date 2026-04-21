from django.urls import path
from .views import home, upload_video, stream_video, clear_events, events_api

urlpatterns = [
    path("", home, name="home"),
    path("upload/", upload_video, name="upload"),
    path("stream/<str:filename>/", stream_video, name="stream_video"),
    path('clear/', clear_events, name='clear_events'),
    path("events-api/", events_api, name="events_api"),
]