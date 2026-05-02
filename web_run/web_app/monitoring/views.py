import os
import csv
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import StreamingHttpResponse, HttpResponseBadRequest, JsonResponse, FileResponse, Http404
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from yolo_lstm_process.engine import generate_processed_frames, generate_processed_frames_camera
from .models import Event
from django.views.decorators.http import require_POST


@login_required(login_url='login')
def events_api(request):
    events = Event.objects.order_by('-timestamp')[:50]

    data = []
    for i, event in enumerate(events, start=1):
        local_time = timezone.localtime(event.timestamp)

        data.append({
            "stt": i,
            "event_type": event.event_type,
            "confidence": round(event.confidence, 2),
            "timestamp": local_time.strftime("%d/%m/%Y %H:%M:%S"),
            "image_url": event.image.url if event.image else "",
            "clip_url": event.clip.url if event.clip else "",
        })

    return JsonResponse({"events": data})


@login_required(login_url='login')
@require_POST
def clear_events(request):
    Event.objects.all().delete()
    request.session.pop("last_video_name", None)
    request.session.pop("last_csv_filename", None)
    return redirect("home")


@login_required(login_url='login')
def home(request):
    events = Event.objects.order_by("-timestamp")
    csv_filename = request.session.get("last_csv_filename")

    return render(request, "home/home.html", {
        "events": events,
        "csv_filename": csv_filename,
    })


@login_required(login_url='login')
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(video_file.name, video_file)

        request.session["last_video_name"] = filename
        request.session["last_csv_filename"] = f"{os.path.splitext(filename)[0]}_lstm_statistics.csv"

        video_url = f"{settings.MEDIA_URL}uploads/{filename}"
        events = Event.objects.order_by("-timestamp")

        return render(request, "home/home.html", {
            "input_video": video_url,
            "stream_name": filename,
            "start_stream": True,
            "events": events,
            "csv_filename": request.session.get("last_csv_filename"),
        })

    events = Event.objects.order_by("-timestamp")
    return render(request, "home/home.html", {
        "events": events,
        "csv_filename": request.session.get("last_csv_filename"),
    })


@login_required(login_url='login')
def stream_video(request, filename):
    input_path = os.path.join(settings.MEDIA_ROOT, "uploads", filename)

    if not os.path.exists(input_path):
        return HttpResponseBadRequest("Video không tồn tại.")

    return StreamingHttpResponse(
        generate_processed_frames(input_path),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


@login_required(login_url='login')
def stream_camera(request):
    return StreamingHttpResponse(
        generate_processed_frames_camera(0),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


@login_required(login_url='login')
def view_csv(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, "test_outputs", filename)

    if not os.path.exists(file_path):
        return render(request, "home/view_csv.html", {
            "error": "File thống kê chưa được tạo xong hoặc không tồn tại.",
            "filename": filename
        })

    data = []
    headers = []

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            headers = []
        else:
            for row in reader:
                data.append(row)

    return render(request, "home/view_csv.html", {
        "headers": headers,
        "data": data,
        "filename": filename
    })
    
@login_required(login_url='login')
def start_camera_page(request):
    camera_csv = f"camera_live_lstm_statistics.csv"
    request.session["last_csv_filename"] = camera_csv

    events = Event.objects.order_by("-timestamp")
    return render(request, "home/home.html", {
        "start_camera": True,
        "events": events,
        "csv_filename": request.session.get("last_csv_filename"),
    })

@login_required(login_url='login')
def start_camera(request):
    request.session["last_csv_filename"] = "camera_live_lstm_statistics.csv"
    return JsonResponse({"ok": True})

def download_csv(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, "test_outputs", filename)

    if not os.path.exists(file_path):
        raise Http404("Không tìm thấy file CSV")

    return FileResponse(
        open(file_path, "rb"),
        as_attachment=True,
        filename=filename
    )