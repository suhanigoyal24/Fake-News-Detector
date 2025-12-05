from django.db import models

class Review(models.Model):
    name = models.CharField(max_length=100, blank=True)
    review = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name}: {self.review[:20]}"
