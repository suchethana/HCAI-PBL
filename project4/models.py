from django.db import models

class UserStudyData(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    group = models.CharField(max_length=1)
    ratings_data = models.JSONField()
    perceived_accuracy = models.IntegerField(null=True, blank=True)
    trust = models.IntegerField(null=True, blank=True)
    qualitative_feedback = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)