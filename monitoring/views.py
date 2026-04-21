import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import StreamingHttpResponse, HttpResponseBadRequest

from yolo_lstm_process.engine import generate_processed_frames
from .models import Event

from monitoring.models import Event
from django.shortcuts import redirect
from django.http import JsonResponse


def events_api(request):
    events = Event.objects.order_by('-timestamp')[:50]

    data = []
    for i, event in enumerate(events, start=1):
        data.append({
            "stt": i,
            "event_type": event.event_type,
            "confidence": round(event.confidence, 2),
            "timestamp": event.timestamp.strftime("%d/%m/%Y %H:%M:%S"),
            "image_url": event.image.url if event.image else "",
            "clip_url": event.clip.url if event.clip else "",
        })

    return JsonResponse({"events": data})

def clear_events(request):
    Event.objects.all().delete()
    return redirect('home')


def home(request):
    events = Event.objects.order_by('-timestamp')
    return render(request, "home/home.html", {
        "events": events
    })


def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(video_file.name, video_file)

        video_url = f"{settings.MEDIA_URL}uploads/{filename}"
        events = Event.objects.order_by('-timestamp')

        return render(request, "home/home.html", {
            "input_video": video_url,
            "stream_name": filename,
            "start_stream": True,
            "events": events,
        })

    events = Event.objects.order_by('-timestamp')
    return render(request, "home/home.html", {
        "events": events
    })


def stream_video(request, filename):
    input_path = os.path.join(settings.MEDIA_ROOT, "uploads", filename)

    if not os.path.exists(input_path):
        return HttpResponseBadRequest("Video không tồn tại.")

    return StreamingHttpResponse(
        generate_processed_frames(input_path),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )