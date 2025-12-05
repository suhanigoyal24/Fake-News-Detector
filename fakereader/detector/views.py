from rest_framework import generics
from .models import Review
from .serializers import ReviewSerializer

class ReviewListCreateAPIView(generics.ListCreateAPIView):
    queryset = Review.objects.all().order_by('-created_at')  # latest first
    serializer_class = ReviewSerializer
