from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_video, name="upload_video"),
    path("stream/<str:filename>/", views.stream_video, name="stream_video"),
    path("stream-camera/", views.stream_camera, name="stream_camera"),
    path("start-camera/", views.start_camera, name="start_camera"),
    path("events-api/", views.events_api, name="events_api"),
    path("clear-events/", views.clear_events, name="clear_events"),

    path("view/csv/<str:filename>/", views.view_csv, name="view_csv"),
    path("download/csv/<str:filename>/", views.download_csv, name="download_csv"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    

