from django.db import models

class Event(models.Model):
    EVENT_TYPES = [
        ('violence', 'Violence'),
        ('weapon', 'Weapon'),
    ]

    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='alerts/images/', null=True, blank=True)
    clip = models.FileField(upload_to='alerts/clips/', null=True, blank=True)
    email_sent = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.event_type} - {self.timestamp}"